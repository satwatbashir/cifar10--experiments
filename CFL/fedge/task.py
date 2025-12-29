# fedge/task.py  — CIFAR-10 ONLY

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets.partitioner import DirichletPartitioner
import datasets  # Hugging Face Datasets
import os
from datasets import DownloadConfig
import random

# Avoid any network checks
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
_DL_CFG = DownloadConfig(local_files_only=True)

# Persistent, per-actor cache of constructed loaders
_DATA_CACHE = {}
_LABELS_CACHE = {}

# Exposed to the rest of the app
DATA_FLAGS = ["cifar10"]

# ───────────────────────── Model (FedProx-style LeNet for apples-to-apples) ─────────────────────────
class Net(nn.Module):
    def __init__(self, in_ch: int = 3, img_h: int = 32, img_w: int = 32, n_class: int = 10):
        super().__init__()
        # Simple LeNet-style CNN used in FedProx baseline
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

# ───────────────────────── CIFAR-10 data ─────────────────────────
CIFAR_TFM = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

class HFDatasetAsTorch(Dataset):
    """Tiny wrapper to apply torchvision-style transforms over HF datasets."""
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"] if "image" in ex else ex["img"]
        lbl = int(ex["label"])
        if self.transform:
            img = self.transform(img)
        return img, lbl

# One module-level partitioner reused across calls (Flower style)
partitioner = DirichletPartitioner(
    num_partitions=1,
    partition_by="label",
    alpha=0.3,              # default; will be overridden per-call
    min_partition_size=10,
    self_balancing=False,
    shuffle=True,
    seed=0,
)

def _split_indices_train_val(indices: np.ndarray, val_frac: float = 0.30, seed: int = 0):
    rng = np.random.default_rng(seed=seed)
    idx = rng.permutation(indices)
    k = max(1, int(val_frac * len(idx)))
    val_idx = idx[:k]
    train_idx = idx[k:]
    return train_idx.tolist(), val_idx.tolist()

# ─────────────────── load_data ────────────────────────────
def load_data(
    dataset_flag: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    *,
    dirichlet_alpha: float = 0.5,
    base_seed: int = 0,
):
    """
    Return (trainloader, valloader, n_classes) for a Dirichlet split of CIFAR-10.

    - dataset_flag must be "cifar10".
    - partition_id: integer in [0..num_partitions-1].
    - num_partitions: number of clients.
    - batch_size: DataLoader batch size.
    """
    if dataset_flag.lower() != "cifar10":
        raise ValueError("This loader only supports dataset_flag='cifar10'.")

    # key includes split settings so we don't rebuild per round
    key = (
        "cifar10",
        int(partition_id),
        int(num_partitions),
        int(batch_size),
        float(dirichlet_alpha),
        int(base_seed),
    )
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    # Local-only: never hit the hub; use cached files
    hf = datasets.load_dataset("cifar10", download_config=_DL_CFG)
    train_full = HFDatasetAsTorch(hf["train"], transform=CIFAR_TFM)
    test_full  = HFDatasetAsTorch(hf["test"],  transform=CIFAR_TFM)

    # Cache the tiny labels-only table so we don't "cast" every time
    lab_key = ("cifar10-labels", len(hf["train"]))
    if lab_key not in _LABELS_CACHE:
        labels_np = np.array(hf["train"]["label"], dtype=np.int64)
        hf_label_ds = datasets.Dataset.from_dict({"label": labels_np.tolist()})
        hf_label_ds = hf_label_ds.cast_column("label", datasets.Value("int64"))
        _LABELS_CACHE[lab_key] = (labels_np, hf_label_ds)
    else:
        labels_np, hf_label_ds = _LABELS_CACHE[lab_key]

    # 4) Reconfigure the existing module-level partitioner for this run
    partitioner._num_partitions = num_partitions
    partitioner._alpha = partitioner._initialize_alpha(float(dirichlet_alpha))
    partitioner._partition_by = "label"
    partitioner._min_partition_size = 10
    partitioner._self_balancing = False
    partitioner._shuffle = True
    partitioner._seed = int(base_seed)
    partitioner._rng = np.random.default_rng(seed=int(base_seed))

    # Assign the labels-only dataset if not already set
    if not partitioner.is_dataset_assigned():
        partitioner.dataset = hf_label_ds  # hf_label_ds is HF Dataset with just the int64 "label" column

    # Clear cached mapping and (re)compute indices
    partitioner._partition_id_to_indices = {}
    partitioner._partition_id_to_indices_determined = False
    partitioner._determine_partition_id_to_indices_if_needed()

    # Grab this client's indices (note: keys are ints)
    client_indices = np.array(
        partitioner._partition_id_to_indices[int(partition_id)],
        dtype=np.int64
    )

    # 6) Per-client 90/10 split (train/val) to keep metrics semantics identical
    train_idx, val_idx = _split_indices_train_val(
        client_indices, val_frac=0.30, seed=int(base_seed) + int(partition_id)
    )

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(train_full, val_idx) if len(val_idx) > 0 else Subset(train_full, train_idx[:1])

    # 7) DataLoaders (Windows-friendly: num_workers=0)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    valloader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Cache and return
    out = (trainloader, valloader, 10)
    _DATA_CACHE[key] = out
    return out

# ─────────────────── Centralized loaders for server eval ───────────────────
def load_global_data(batch_size: int = 256):
    hf = datasets.load_dataset("cifar10", download_config=_DL_CFG)
    train_full = HFDatasetAsTorch(hf["train"], transform=CIFAR_TFM)
    test_full  = HFDatasetAsTorch(hf["test"],  transform=CIFAR_TFM)
    trainloader = DataLoader(train_full, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader  = DataLoader(test_full,  batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader, 10

# ─────────────────── Utils ────────────────────────────
def test(net: nn.Module, loader: DataLoader, device: torch.device):
    net.eval()
    net.to(device)
    ce = nn.CrossEntropyLoss(reduction="mean")
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = net(images)
            loss = ce(logits, labels)
            loss_sum += float(loss.item()) * int(labels.shape[0])
            correct += int((logits.argmax(1) == labels).sum().item())
            total += int(labels.shape[0])
    avg_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)
    # Clamp to reasonable ranges
    avg_loss = min(avg_loss, 1e6)
    acc = max(0.0, min(1.0, acc))
    return avg_loss, acc

def get_weights(net: nn.Module):
    return [v.detach().cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, weights):
    keys = list(net.state_dict().keys())
    state_dict = OrderedDict({k: torch.tensor(w) for k, w in zip(keys, weights)})
    net.load_state_dict(state_dict, strict=True)

def set_global_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and Torch for reproducibility."""
    try:
        seed = int(seed)
    except Exception:
        seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
