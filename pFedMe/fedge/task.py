# fedge/task.py

from collections import OrderedDict
from typing import Tuple, Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10
from flwr_datasets.partitioner import DirichletPartitioner
import datasets  # HuggingFace

# Only CIFAR-10 now
DATA_FLAGS = ["cifar10"]

# ───────────────────────── Seeding ──────────────────────────
def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility without forcing strict determinism.

    This mirrors performance-friendly settings: deterministic algorithms off
    (to allow fast kernels), cudnn benchmarking on.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prefer speed over strict determinism for this project
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

# ───────────────────────── Net ──────────────────────────
class Net(nn.Module):
    def __init__(self, in_ch: int = 3, img_h: int = 32, img_w: int = 32, n_class: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # infer flatten size for safety
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(torch.zeros(1, in_ch, img_h, img_w))))
            x = self.pool(F.relu(self.conv2(x)))
            flat = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

# ─────────────────── CIFAR-10 helpers ───────────────────
CIFAR_TFM = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

def _require_alpha(alpha: Optional[float]) -> float:
    if alpha is None:
        raise ValueError(
            "Dirichlet alpha must be provided via run_config['dirichlet_alpha'] in pyproject.toml"
        )
    return float(alpha)

def _dl(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    # OOM-safe defaults; pin_memory only if CUDA is on
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )

# ─────────────────── Central loaders (optional helper) ───────────────────
_central_cache = {"ready": False}
def get_central_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader, int, int, int, int]:
    """
    Optional helper for centralized evaluation.
    Returns (trainloader, testloader, in_ch, H, W, num_classes).
    """
    global _central_cache
    if _central_cache.get("ready", False):
        return (_central_cache["trainloader"], _central_cache["testloader"],
                _central_cache["in_ch"], _central_cache["H"], _central_cache["W"],
                _central_cache["num_classes"])

    tr = CIFAR10(root="./data", train=True,  download=True, transform=CIFAR_TFM)
    te = CIFAR10(root="./data", train=False, download=True, transform=CIFAR_TFM)

    tl = _dl(tr, batch_size=batch_size, shuffle=False)
    vl = _dl(te, batch_size=batch_size, shuffle=False)

    sample, _ = next(iter(tl))
    _, in_ch, H, W = sample.shape

    _central_cache.update({
        "ready": True,
        "trainloader": tl,
        "testloader": vl,
        "in_ch": int(in_ch),
        "H": int(H),
        "W": int(W),
        "num_classes": 10,
    })
    return tl, vl, int(in_ch), int(H), int(W), 10

# ─────────────────── load_data (separate Dirichlet for train/test) ───────────────────
def load_data(
    dataset_flag: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    alpha: Optional[float] = None,
    seed: int = 0,
):
    """
    Return (trainloader, testloader, n_classes) for a Dirichlet split of CIFAR-10.
    'alpha' MUST be passed from run_config (no hard-coding).
    Train and test are partitioned independently by label via Dirichlet.
    """
    if dataset_flag.lower() != "cifar10":
        raise ValueError("This loader only supports dataset_flag='cifar10'")

    alpha = _require_alpha(alpha)

    # Seed everything for deterministic partitioning given run seed
    set_global_seed(int(seed))

    # 1) Base datasets
    train_full = CIFAR10(root="./data", train=True,  download=True, transform=CIFAR_TFM)
    test_full  = CIFAR10(root="./data", train=False, download=True, transform=CIFAR_TFM)

    # 2) Build label-only HF datasets
    train_labels_np = np.asarray(train_full.targets, dtype=np.int64)
    test_labels_np  = np.asarray(test_full.targets,  dtype=np.int64)
    hf_train = datasets.Dataset.from_dict({"label": train_labels_np.tolist()}).cast_column("label", datasets.Value("int64"))
    hf_test  = datasets.Dataset.from_dict({"label": test_labels_np .tolist()}).cast_column("label", datasets.Value("int64"))

    # 3) Dirichlet partitioners
    train_part = DirichletPartitioner(
        num_partitions=num_partitions, partition_by="label",
        alpha=alpha, min_partition_size=10, self_balancing=False, shuffle=True, seed=int(seed)
    )
    train_part.dataset = hf_train
    train_part._determine_partition_id_to_indices_if_needed()
    tr_idx = train_part._partition_id_to_indices[partition_id]

    test_part = DirichletPartitioner(
        num_partitions=num_partitions, partition_by="label",
        alpha=alpha, min_partition_size=5, self_balancing=False, shuffle=True, seed=int(seed)
    )
    test_part.dataset = hf_test
    test_part._determine_partition_id_to_indices_if_needed()
    te_idx = test_part._partition_id_to_indices[partition_id]

    # 4) Final loaders
    trainloader = _dl(Subset(train_full, tr_idx), batch_size=batch_size, shuffle=True)
    testloader  = _dl(Subset(test_full,  te_idx), batch_size=batch_size, shuffle=False)

    return trainloader, testloader, 10  # n_classes

# ─────────────────── load_data_with_larger_test (single Dirichlet, then 80/20 split) ───────────────────
def load_data_with_larger_test(
    dataset_flag: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    alpha: Optional[float] = None,
    seed: int = 0,
):
    """
    Return (trainloader, testloader, n_classes) where clients get:
      - A single Dirichlet partition over combined (train+test) CIFAR-10 indices.
      - The client's shard is split 80/20 into train/test → client evaluation uses same distribution as its training shard.
    'alpha' MUST be passed from run_config (no hard-coding).
    """
    if dataset_flag.lower() != "cifar10":
        raise ValueError("This loader only supports dataset_flag='cifar10'")

    alpha = _require_alpha(alpha)

    # Seed everything for deterministic partitioning given run seed
    set_global_seed(int(seed))

    # 1) Load base datasets
    train_full = CIFAR10(root="./data", train=True,  download=True, transform=CIFAR_TFM)
    test_full  = CIFAR10(root="./data", train=False, download=True, transform=CIFAR_TFM)

    # 2) Combine labels from both splits for partitioning
    train_labels_np = np.asarray(train_full.targets, dtype=np.int64)
    test_labels_np  = np.asarray(test_full.targets,  dtype=np.int64)
    combined_labels = np.concatenate([train_labels_np, test_labels_np])

    hf_combined = datasets.Dataset.from_dict({"label": combined_labels.tolist()}).cast_column("label", datasets.Value("int64"))

    # 3) Dirichlet partition over combined set
    combined_part = DirichletPartitioner(
        num_partitions=num_partitions, partition_by="label",
        alpha=alpha, min_partition_size=60, self_balancing=False, shuffle=True, seed=int(seed)
    )
    combined_part.dataset = hf_combined
    combined_part._determine_partition_id_to_indices_if_needed()
    client_indices = np.array(combined_part._partition_id_to_indices[partition_id], dtype=np.int64)

    # 4) 80/20 split of client shard (deterministic per client)
    if client_indices.size == 0:
        # Edge case guard (shouldn't happen with CIFAR-10 + above min sizes)
        empty = Subset(train_full, [])
        return _dl(empty, batch_size, True), _dl(empty, batch_size, False), 10

    rng = np.random.default_rng(int(seed) + int(partition_id))
    rng.shuffle(client_indices)

    split_point = int(0.8 * len(client_indices))
    client_train_indices = client_indices[:split_point]
    client_test_indices  = client_indices[split_point:]

    # 5) Map combined indices back to original train/test ranges
    train_size = len(train_full)

    # train side
    tr_mask_from_train = client_train_indices < train_size
    tr_mask_from_test  = ~tr_mask_from_train
    final_train_indices            = client_train_indices[tr_mask_from_train]
    final_train_indices_from_test  = client_train_indices[tr_mask_from_test] - train_size

    # test side
    te_mask_from_train = client_test_indices < train_size
    te_mask_from_test  = ~te_mask_from_train
    final_test_indices           = client_test_indices[te_mask_from_train]
    final_test_indices_from_test = client_test_indices[te_mask_from_test] - train_size

    # 6) Build per-client train/test datasets (union from train/test pools)
    train_parts = []
    test_parts  = []

    if final_train_indices.size > 0:
        train_parts.append(Subset(train_full, final_train_indices))
    if final_train_indices_from_test.size > 0:
        train_parts.append(Subset(test_full, final_train_indices_from_test))

    if final_test_indices.size > 0:
        test_parts.append(Subset(train_full, final_test_indices))
    if final_test_indices_from_test.size > 0:
        test_parts.append(Subset(test_full, final_test_indices_from_test))

    combined_train = ConcatDataset(train_parts) if train_parts else Subset(train_full, [])
    combined_test  = ConcatDataset(test_parts)  if test_parts  else Subset(test_full,  [])

    # 7) Final loaders
    trainloader = _dl(combined_train, batch_size=batch_size, shuffle=True)
    testloader  = _dl(combined_test,  batch_size=batch_size, shuffle=False)

    return trainloader, testloader, 10  # n_classes

# ─────────────────── train / test / weights ───────────────────
def train(net: nn.Module, loader: DataLoader, epochs: int, device: torch.device) -> float:
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    ce  = nn.CrossEntropyLoss()
    net.train()
    last_loss = 0.0
    for _ in range(int(epochs)):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if labels.ndim > 1:
                labels = labels.squeeze()
            labels = labels.long()
            opt.zero_grad()
            loss = ce(net(images), labels)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
    return float(last_loss)

def test(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    net.to(device)
    ce  = nn.CrossEntropyLoss()
    net.eval()
    total_samples, correct, loss_sum, processed_batches = 0, 0, 0.0, 0

    # Handle empty dataloader
    try:
        _len = len(loader)
    except TypeError:
        _len = 0
    if _len == 0:
        return (0.0, 0.0)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            labels = labels.long().view(-1)
            if labels.size(0) != images.size(0):
                continue
            out = net(images)
            bs = int(labels.size(0))
            loss_sum += float(ce(out, labels).item()) * bs
            correct  += int((out.argmax(1) == labels).sum().item())
            total_samples += bs
            processed_batches += 1

    if total_samples == 0 or processed_batches == 0:
        return (0.0, 0.0)

    return (loss_sum / max(1, total_samples), correct / max(1, total_samples))

def get_weights(net: nn.Module):
    return [v.detach().cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, w):
    state_dict = OrderedDict((k, torch.tensor(v)) for k, v in zip(net.state_dict().keys(), w))
    net.load_state_dict(state_dict, strict=True)
