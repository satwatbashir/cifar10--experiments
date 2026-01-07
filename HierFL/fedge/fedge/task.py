# fedge/task.py

from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import datasets as tv_datasets, transforms
import numpy as np
from flwr_datasets.partitioner import DirichletPartitioner
import datasets as hf_datasets  # HuggingFace
import functools
import contextlib
import os
import toml
from pathlib import Path

# Load Dirichlet concentration parameter for clients from config
task_script_dir = Path(__file__).resolve().parent
task_project_root = task_script_dir.parent
cfg = toml.load(task_project_root / "pyproject.toml")
hier = cfg["tool"]["flwr"]["hierarchy"]
ALPHA_CLIENT = hier.get("alpha_client", 0.3)
APP_CFG = cfg["tool"]["flwr"]["app"]["config"]
DEFAULT_BATCH_SIZE = int(APP_CFG.get("batch_size", 64))
DEFAULT_SEED = int(APP_CFG.get("seed", 42))

# Decorator to suppress verbose prints and progress bars
def suppress_output(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return func(*args, **kwargs)
    return wrapper

# ───────────────────── Global seeding ─────────────────────
def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prefer performance; set deterministic=False to match FedProx default
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

# ───────────────────────── ResNet-18 for CIFAR-10 ──────────────────────────
# Adapted for 32x32 images: uses 3x3 conv with stride 1, no initial maxpool
# Reference: NIID-Bench (ICDE 2022) settings with ResNet-18 (~11.2M params)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        # CIFAR-10 adaptation: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    """ResNet-18 for CIFAR-10 (~11.2M parameters)"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# ───────────────────────── LeNet (NIID-Bench CNN) ──────────────────────────
# Standard NIID-Bench model for CIFAR-10 (~62K parameters)
# Reference: Li et al. (2022) ICDE - NIID-Bench benchmark
# Expected accuracy: 65-70% with Dir(0.5), 10 clients, 50 rounds

class Net(nn.Module):
    """LeNet-style CNN for CIFAR-10 (NIID-Bench standard)."""
    def __init__(self, in_ch: int = 3, img_h: int = 32, img_w: int = 32, n_class: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, kernel_size=5)     # 3x32x32 -> 6x28x28
        self.pool  = nn.MaxPool2d(2, 2)                     # -> 6x14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        # -> 16x10x10 -> pool -> 16x5x5
        # compute flatten dim dynamically
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(torch.zeros(1, in_ch, img_h, img_w))))
            x = self.pool(F.relu(self.conv2(x)))
            flat = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ─────────────── CIFAR-10 transforms (Flower style) ───────────────
# Standard mean/std for CIFAR-10
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

TFM_TRAIN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])

TFM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])

# ─────────────── Module-level dataset cache ───────────────
# Cache full datasets to avoid reloading for each client (memory optimization)
_CIFAR10_CACHE: dict = {"train": None, "test": None}

def _get_cached_cifar10(train: bool = True):
    """Return cached CIFAR-10 dataset, loading once on first call."""
    key = "train" if train else "test"
    if _CIFAR10_CACHE[key] is None:
        transform = TFM_TRAIN if train else TFM_TEST
        _CIFAR10_CACHE[key] = tv_datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
    return _CIFAR10_CACHE[key]

# ─────────────────────── load_data ───────────────────────
@suppress_output
def load_data(
    dataset_flag: str,
    partition_id: int,
    num_partitions: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    alpha: Optional[float] = None,
    seed: int = DEFAULT_SEED,
    indices: Optional[List[int]] = None,
    indices_test: Optional[List[int]] = None,
):
    """
    Return (trainloader, testloader, n_classes) for a Dirichlet split of CIFAR-10.

    - dataset_flag must be "cifar10".
    - partition_id: integer in [0..num_partitions-1].
    - num_partitions: how many clients per leaf server.
    - batch_size: DataLoader batch size (baseline 32).
    """
    if dataset_flag.lower() != "cifar10":
        raise ValueError("This loader only supports dataset_flag='cifar10'")

    # 1) Load full CIFAR-10 (cached at module level for memory efficiency)
    train_full = _get_cached_cifar10(train=True)
    test_full  = _get_cached_cifar10(train=False)

    # 2) Build HF labels datasets for partitioner (labels only)
    train_labels = [int(y) for y in train_full.targets]  # 50k
    test_labels  = [int(y) for y in test_full.targets]   # 10k
    hf_train = hf_datasets.Dataset.from_dict({"label": train_labels}).cast_column("label", hf_datasets.Value("int64"))
    hf_test  = hf_datasets.Dataset.from_dict({"label":  test_labels}).cast_column("label",  hf_datasets.Value("int64"))

    # Ensure deterministic partitioning
    set_global_seed(seed)
    used_alpha = float(alpha if alpha is not None else ALPHA_CLIENT)

    # If explicit *indices* provided (HierFL hierarchical split), use them directly.
    if indices is not None:
        client_indices = indices
    else:
        # 3) Partition train indices with DirichletPartitioner
        train_partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=used_alpha,
            min_partition_size=100,
            self_balancing=False,
            shuffle=True,
            seed=seed,
        )
        train_partitioner.dataset = hf_train
        train_partitioner._determine_partition_id_to_indices_if_needed()
        client_indices = train_partitioner._partition_id_to_indices[partition_id]

    # 4) Wrap with Subset + DataLoader
    trainloader = DataLoader(Subset(train_full, client_indices), batch_size=batch_size, shuffle=True, num_workers=0)

    # 5) Build test loader: prefer explicit indices_test, else derive as before
    if indices_test is not None:
        testloader = DataLoader(Subset(test_full, indices_test), batch_size=batch_size, shuffle=False, num_workers=0)
    elif indices is not None:
        test_partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=used_alpha,
            min_partition_size=20,
            self_balancing=False,
            shuffle=True,
            seed=seed,
        )
        test_partitioner.dataset = hf_test
        test_partitioner._determine_partition_id_to_indices_if_needed()
        test_indices = test_partitioner._partition_id_to_indices[partition_id]
        testloader = DataLoader(Subset(test_full, test_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        testloader = DataLoader(test_full, batch_size=batch_size, shuffle=False, num_workers=0)

    # 6) CIFAR-10 classes
    n_class = 10
    return trainloader, testloader, n_class

# ─────────────────── train / test / weights ───────────────────
def train(net: nn.Module, loader: DataLoader, epochs: int, device: torch.device, lr: float = 0.01):
    """Train for `epochs` full passes over the dataset.

    Changed from step-based to epoch-based training to match FedProx.
    Each epoch iterates over all batches in the DataLoader.
    """
    net.to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    ce = nn.CrossEntropyLoss()
    net.train()

    loss = 0.0
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if labels.ndim > 1:
                labels = labels.squeeze(-1)
            labels = labels.long()
            opt.zero_grad()
            loss = ce(net(images), labels)
            loss.backward()
            opt.step()
    return float(loss)

def test(net: nn.Module, loader: DataLoader, device: torch.device):
    net.to(device)
    ce  = nn.CrossEntropyLoss()
    net.eval()
    total_samples, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if labels.ndim > 1:
                labels = labels.squeeze(-1)
            # Cast to required dtype for CrossEntropyLoss
            labels = labels.long()
            out = net(images)
            # Accumulate total loss weighted by batch size (sample-wise average)
            loss_sum += ce(out, labels).item() * labels.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    if total_samples == 0:
        return 0.0, 0.0
    return (loss_sum / max(1, total_samples), correct / max(1, total_samples))

def get_weights(net: nn.Module):
    return [v.cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, w):
    net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), w)}))
