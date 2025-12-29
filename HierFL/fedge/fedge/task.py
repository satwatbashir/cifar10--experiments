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
DEFAULT_BATCH_SIZE = int(APP_CFG.get("batch_size", 32))
DEFAULT_SEED = int(APP_CFG.get("seed", 0))

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

# ───────────────────────── Net ──────────────────────────
class Net(nn.Module):
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

    # 1) Load full CIFAR-10 (download once)
    train_full = tv_datasets.CIFAR10(root="./data", train=True, download=True, transform=TFM_TRAIN)
    test_full  = tv_datasets.CIFAR10(root="./data", train=False, download=True, transform=TFM_TEST)

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
def train(net: nn.Module, loader: DataLoader, epochs: int, device: torch.device):
    """Local training on *epochs* mini-batch updates (matches HierFL).

    In the original HierFL implementation the hyper-parameter `num_local_update`
    counts **optimizer steps**, not full passes over the dataset.  This revised
    loop keeps the same semantics: each iteration consumes exactly one batch; if
    we reach the end of the loader we simply restart it.
    """
    net.to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    ce = nn.CrossEntropyLoss()
    net.train()

    data_iter = iter(loader)
    step = 0
    loss = 0.0
    while step < epochs:  # epochs == #updates
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        if labels.ndim > 1:
            labels = labels.squeeze(-1)
        # Cast to required dtype for CrossEntropyLoss
        labels = labels.long()
        opt.zero_grad()
        loss = ce(net(images), labels)
        loss.backward()
        opt.step()
        scheduler.step()  # decay LR every update, as in HierFL
        step += 1
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
