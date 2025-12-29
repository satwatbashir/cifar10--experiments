# fedge/task.py  – CIFAR-10 only implementation
# -----------------------------------------------------------
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence
from collections import OrderedDict

import datasets as hf_datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import toml

logger = logging.getLogger(__name__)

# ───────────────────────────── Config helpers ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _get_hier_cfg() -> Dict:
    return toml.load(PROJECT_ROOT / "pyproject.toml")["tool"]["flwr"]["hierarchy"]

# ─────────────────────────────  Transforms  ────────────────────────────────
# CIFAR-10 transforms with proper normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

TRAIN_TRANSFORM = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

TEST_TRANSFORM = Compose([
    ToTensor(),
    Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ─────────────────────────────  Net  ───────────────────────────────────────
class Net(nn.Module):
    """CNN for CIFAR-10 (3x32x32 → 10 classes)."""

    def __init__(self, in_ch: int = 3, img_h: int = 32, img_w: int = 32, n_class: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size for CIFAR-10 (3x32x32)
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(torch.zeros(1, in_ch, img_h, img_w))))
            x = self.pool(F.relu(self.conv2(x)))
            flat = x.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ──────────────────────── Data utilities ───────────────────────────────────
# Global dataset cache to prevent repeated downloads
_DATASET_CACHE = {}

def _dataset_root() -> Path:
    """Return absolute dataset root path.
    We intentionally fix this to PROJECT_ROOT/data to avoid ambiguous CWD-relative paths.
    """
    return (PROJECT_ROOT / "data").resolve()

def _cleanup_cifar10(root: Path) -> None:
    """Remove corrupted CIFAR-10 artifacts to force a clean re-download.
    Deletes the archive and extracted folder if present.
    """
    try:
        tar_path = root / "cifar-10-python.tar.gz"
        extracted_dir = root / "cifar-10-batches-py"
        if tar_path.exists():
            logger.warning(f"Removing existing archive: {tar_path}")
            tar_path.unlink()
        if extracted_dir.exists():
            logger.warning(f"Removing extracted directory: {extracted_dir}")
            shutil.rmtree(extracted_dir)
    except Exception as e:
        logger.error(f"Failed to cleanup CIFAR-10 artifacts under {root}: {e}")


def get_cifar10_dataset(train: bool = True):
    """Load CIFAR-10 dataset using cached HuggingFace datasets with retry logic"""
    cache_key = f"cifar10_{'train' if train else 'test'}"
    
    # Return cached dataset if available
    if cache_key in _DATASET_CACHE:
        logger.debug(f"📋 Using cached CIFAR-10 {'train' if train else 'test'} dataset ({len(_DATASET_CACHE[cache_key])} samples)")
        return _DATASET_CACHE[cache_key]
    
    # Dataset loading log removed for brevity
    
    # Retry logic for network robustness
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            split = "train" if train else "test"
            
            # First try: normal download
            if attempt == 0:
                dataset = hf_datasets.load_dataset("cifar10", split=split, trust_remote_code=True)
            # Second try: force redownload
            elif attempt == 1:
                logger.warning(f"Retry {attempt}: Attempting force redownload")
                dataset = hf_datasets.load_dataset("cifar10", split=split, trust_remote_code=True, download_mode="force_redownload")
            # Final try: use cached version only
            else:
                logger.warning(f"Retry {attempt}: Attempting cached version only")
                dataset = hf_datasets.load_dataset("cifar10", split=split, trust_remote_code=True, download_mode="reuse_cache_if_exists")
            
            # Cache the dataset
            _DATASET_CACHE[cache_key] = dataset
            return dataset
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"❌ All {max_retries} attempts failed to load CIFAR-10 from HuggingFace")
                raise RuntimeError(f"CIFAR-10 loading failed after {max_retries} attempts: {e}") from e

def get_cifar10_test_loader(batch_size: int = 32) -> DataLoader:
    """Get CIFAR-10 test loader for global evaluation."""
    hf_dataset = get_cifar10_dataset(train=False)
    test_dataset = HuggingFaceCifarDataset(hf_dataset)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class HuggingFaceCifarDataset(Dataset):
    """Wrapper for HuggingFace CIFAR-10 dataset."""

    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        # Extract labels as numpy array for partitioning
        self.labels = np.array([item['label'] for item in hf_dataset], dtype=np.int64)
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        # Convert PIL image to tensor
        image = item['img']
        label = item['label']
        
        # Apply transforms
        transform = get_transform(train=False)  # Use test transforms for consistency
        if transform:
            image = transform(image)
        
        return image, label

class CifarDataset(Dataset):
    """Wrapper for CIFAR-10 dataset to expose labels as numpy array."""

    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        # Extract labels as numpy array for partitioning
        self.labels = np.array([item['label'] for item in hf_dataset], dtype=np.int64)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        item = self.hf_dataset[idx]
        # Convert PIL image to tensor and apply transforms
        image = item['img']
        label = item['label']
        
        # Apply transforms
        transform = get_transform(train=True)  # Use training transforms
        if transform:
            image = transform(image)
        
        return image, label


def _to_plain_ints(arr: Sequence) -> List[int]:
    return [int(x if not hasattr(x, "item") else x.item()) for x in arr]


# Global partition cache to prevent repeated JSON loading
_PARTITION_CACHE = {}
_PARTITION_FILE_CACHE = None

def _load_partition_indices(server_id: int, partition_id: int) -> List[int]:
    """Load pre-created partition indices from JSON file with caching."""
    import json
    import os
    
    global _PARTITION_FILE_CACHE
    
    parts_json = os.getenv("PARTITIONS_JSON")
    if not parts_json:
        parts_json = str(PROJECT_ROOT / "rounds" / "partitions.json")
    
    # Cache the entire partition file to avoid repeated JSON parsing
    if _PARTITION_FILE_CACHE is None:
        logger.debug(f"📂 Loading partition file {parts_json} (first time only)")
        try:
            with open(parts_json, "r") as f:
                _PARTITION_FILE_CACHE = json.load(f)
            logger.info(f"✅ Cached partition file with {len(_PARTITION_FILE_CACHE)} servers")
        except FileNotFoundError:
            logger.error(f"❌ Partition file not found: {parts_json}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading partition file {parts_json}: {e}")
            return []
    
    # Check partition cache first
    cache_key = f"s{server_id}_c{partition_id}"
    if cache_key in _PARTITION_CACHE:
        logger.debug(f"🎯 Using cached partition for server={server_id}, client={partition_id}")
        return _PARTITION_CACHE[cache_key]
    
    try:
        indices = _PARTITION_FILE_CACHE[str(server_id)][str(partition_id)]
        _PARTITION_CACHE[cache_key] = indices
        # Reduced verbosity: only log first partition load per server
        if partition_id == 0:
            logger.info(f"✅ Loading partitions for server {server_id} ({len(indices)} samples for client 0)")
        return indices
        
    except KeyError as e:
        logger.error(f"❌ Partition key not found: server={server_id}, client={partition_id}")
        logger.error(f"Available servers: {list(_PARTITION_FILE_CACHE.keys()) if _PARTITION_FILE_CACHE else 'unknown'}")
        return []
    except Exception as e:
        logger.error(f"❌ Unexpected error loading partition {server_id}/{partition_id}: {e}")
        return []


# Remove _balanced_sampler - not needed for CIFAR-10


def load_data(
    dataset_flag: str,
    partition_id: int,
    num_partitions: int,
    *,
    batch_size: int = 16,
    indices: Optional[List[int]] = None,
    server_id: Optional[int] = None,
):
    dataset_flag = dataset_flag.lower()
    if dataset_flag != "cifar10":
        raise ValueError("dataset_flag must be 'cifar10'")

    # Load CIFAR-10 datasets from HuggingFace
    train_hf = get_cifar10_dataset(train=True)
    test_hf = get_cifar10_dataset(train=False)

    # Wrap in CifarDataset for label extraction and transforms
    train_wrapped = CifarDataset(train_hf)
    test_wrapped = HuggingFaceCifarDataset(test_hf)  # Use test transforms
    
    if indices is not None:
        idx_train = indices
    else:
        idx_train = _load_partition_indices(server_id or 0, partition_id)
    
    # For test set, map training indices to valid test indices proportionally
    train_size = len(train_wrapped)
    test_size = len(test_wrapped)
    
    if train_size == 0 or test_size == 0:
        logger.error(f"Empty dataset: train_size={train_size}, test_size={test_size}")
        idx_test = []
    else:
        # Map each training index to corresponding test index proportionally
        idx_test = [min(int(idx * test_size / train_size), test_size - 1) for idx in idx_train]
        # Remove duplicates while preserving order
        seen = set()
        idx_test = [x for x in idx_test if not (x in seen or seen.add(x))]
        
        logger.debug(f"Mapped {len(idx_train)} train indices to {len(idx_test)} test indices")

    # Validate indices are within bounds
    if idx_train and max(idx_train) >= train_size:
        logger.error(f"Train index out of bounds: max={max(idx_train)}, train_size={train_size}")
        idx_train = [idx for idx in idx_train if idx < train_size]
        logger.warning(f"Filtered to {len(idx_train)} valid train indices")
    
    if idx_test and max(idx_test) >= test_size:
        logger.error(f"Test index out of bounds: max={max(idx_test)}, test_size={test_size}")
        idx_test = [idx for idx in idx_test if idx < test_size]
        logger.warning(f"Filtered to {len(idx_test)} valid test indices")

    train_subset = Subset(train_wrapped, idx_train)
    test_subset = Subset(test_wrapped, idx_test)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, 10  # CIFAR-10 has 10 classes


# ─────────────────────── Training / eval utils ─────────────────────────────
def _make_scheduler(opt: torch.optim.Optimizer, sched_type: str, lr: float):
    sched_type = sched_type.lower()
    if sched_type == "cosine":
        total_gr = int(os.getenv("TOTAL_GLOBAL_ROUNDS", "150"))
        return CosineAnnealingLR(opt, T_max=total_gr, eta_min=lr * 0.01)
    gamma = float(os.getenv("LR_GAMMA", "0.95"))
    return StepLR(opt, step_size=1, gamma=gamma)


def train(
    net: nn.Module,
    loader: DataLoader,
    steps: int,
    device: torch.device,
    *,
    lr: Optional[float] = None,
    momentum: Optional[float] = None,
    weight_decay: Optional[float] = None,
    gamma: Optional[float] = None,  # kept for API compatibility
    clip_norm: Optional[float] = None,
    prox_mu: float = 0.0,
    ref_weights: Optional[List[np.ndarray]] = None,
    global_round: int = 0,
    scaffold_enabled: bool = False,
):
    net.to(device)

    # Read required training parameters from TOML - no fallbacks
    if lr is None or momentum is None or weight_decay is None or clip_norm is None:
        cfg = toml.load(PROJECT_ROOT / "pyproject.toml")
        app_config = cfg["tool"]["flwr"]["app"]["config"]
        
        if lr is None:
            lr = app_config["learning_rate"]
        if momentum is None:
            momentum = app_config["momentum"]
        if weight_decay is None:
            weight_decay = app_config["weight_decay"]
        if clip_norm is None:
            clip_norm = app_config["clip_norm"]
    
    wd = weight_decay
    clip_val = clip_norm

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    sched = (
        None
        if global_round < int(os.getenv("WARMUP_ROUNDS", "5"))  # Keep warmup default for now
        else _make_scheduler(opt, os.getenv("SCHEDULER_TYPE", "step"), lr)  # Keep scheduler default
    )

    ce = nn.CrossEntropyLoss()
    ref_tensors = [torch.tensor(w, device=device) for w in ref_weights] if prox_mu and ref_weights else None

    data_iter, running_loss, n_steps = iter(loader), 0.0, 0
    net.train()
    while n_steps < steps:
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, labels = next(data_iter)

        imgs, labels = imgs.to(device), labels.flatten().long().to(device)
        opt.zero_grad()
        loss = ce(net(imgs), labels)

        if prox_mu and ref_tensors:
            loss += (prox_mu / 2.0) * sum((p - w0).pow(2).sum() for p, w0 in zip(net.parameters(), ref_tensors))

        loss.backward()
        if scaffold_enabled and hasattr(net, "_scaffold_manager"):
            net._scaffold_manager.apply_scaffold_correction(net, opt.param_groups[0]["lr"])

        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
        opt.step()

        running_loss += loss.item()
        n_steps += 1

    if sched:
        sched.step()
    return float(running_loss / max(n_steps, 1))


@torch.no_grad()
def test(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    net.to(device).eval()
    ce, loss_sum, correct, total = nn.CrossEntropyLoss(), 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.flatten().long().to(device)
        out = net(imgs)
        loss_sum += ce(out, labels).item()
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
    return loss_sum / len(loader), correct / total


def get_weights(net: nn.Module) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in net.state_dict().values()]


def set_weights(net: nn.Module, weights: Sequence[np.ndarray]):
    net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), weights)}))


def get_transform(train: bool = True):
    """Return transform for CIFAR-10."""
    return TRAIN_TRANSFORM if train else TEST_TRANSFORM
