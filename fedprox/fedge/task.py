# fedge/task.py

from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from flwr_datasets.partitioner import DirichletPartitioner
import datasets as hfds  # HuggingFace Datasets (labels-only for partitioning)

# ───────────────────── Global seeding ─────────────────────
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep non-deterministic CuDNN for speed; flip to deterministic if you need
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

# ───────────────────── Supported flags ─────────────────────
DATA_FLAGS = ["cifar10"]

# ───────────────────────── ResNet-18 for CIFAR-10 ──────────────────────────
# Adapted for 32x32 images: uses 3x3 conv with stride 1, no initial maxpool
# Reference: FedRAD (PMC), 100 rounds, 10 clients, Dir β=0.5
# Expected accuracy: FedAvg 73.42%, FedProx 75.52%

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
        _CIFAR10_CACHE[key] = datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
    return _CIFAR10_CACHE[key]

# ─────────────────────── load_data ───────────────────────
def load_data(dataset_flag: str,
              partition_id: int,
              num_partitions: int,
              batch_size: int = 32,
              alpha: float = 0.3,
              seed: int = 0):
    """
    Return (trainloader, testloader, n_classes) for a Dirichlet split of CIFAR-10.

    Args:
        dataset_flag: must be "cifar10"
        partition_id: integer in [0..num_partitions-1]
        num_partitions: number of clients
        batch_size: dataloader batch size
        alpha: Dirichlet concentration (heterogeneity)
        seed: random seed controlling partitioning and RNG for this run
    """
    if dataset_flag.lower() != "cifar10":
        raise ValueError("This loader only supports dataset_flag='cifar10'")

    # 1) Load full CIFAR-10 (cached at module level for memory efficiency)
    train_full = _get_cached_cifar10(train=True)
    test_full  = _get_cached_cifar10(train=False)

    # 2) Build HF labels datasets for partitioner (labels only)
    train_labels = [int(y) for y in train_full.targets]  # 50k
    test_labels  = [int(y) for y in test_full.targets]   # 10k
    hf_train = hfds.Dataset.from_dict({"label": train_labels}).cast_column("label", hfds.Value("int64"))
    hf_test  = hfds.Dataset.from_dict({"label":  test_labels}).cast_column("label",  hfds.Value("int64"))

    # 3) Partition train/test indices with DirichletPartitioner
    # Ensure deterministic partitioning per run
    set_global_seed(seed)
    train_partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=alpha,
        min_partition_size=100,
        self_balancing=False,
        shuffle=True,
        seed=seed,
    )
    train_partitioner.dataset = hf_train
    train_partitioner._determine_partition_id_to_indices_if_needed()
    train_indices = train_partitioner._partition_id_to_indices[partition_id]

    test_partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=alpha,
        min_partition_size=20,
        self_balancing=False,
        shuffle=True,
        seed=seed,
    )
    test_partitioner.dataset = hf_test
    test_partitioner._determine_partition_id_to_indices_if_needed()
    test_indices = test_partitioner._partition_id_to_indices[partition_id]

    # 4) Wrap with Subset + DataLoader
    trainloader = DataLoader(Subset(train_full, train_indices), batch_size=batch_size, shuffle=True,  num_workers=0)
    testloader  = DataLoader(Subset(test_full,  test_indices),  batch_size=batch_size, shuffle=False, num_workers=0)

    # 5) CIFAR-10 classes
    n_class = 10
    return trainloader, testloader, n_class

# ─────────────────── train / test / weights ───────────────────
def test(net: nn.Module, loader: DataLoader, device: torch.device):
    net.to(device)
    ce  = nn.CrossEntropyLoss()
    net.eval()
    total, correct, loss_sum = 0, 0, 0.0
    if len(loader) == 0:
        return 0.0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().view(-1)
            if labels.size(0) != images.size(0):
                continue
            out = net(images)
            loss_sum += ce(out, labels).item() * labels.size(0)

            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return (loss_sum / max(1, total), correct / max(1, total))


def get_weights(net: nn.Module):
    return [v.cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, w):
    net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), w)}))
