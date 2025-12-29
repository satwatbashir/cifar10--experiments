"""Utility functions for dynamic server clustering using symmetric KL divergence.

This module is intentionally self-contained so it can be imported by the
cloud-level strategy without pulling in any Flower-specific dependencies.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from fedge.task import Net, get_cifar10_test_loader, set_weights  # type: ignore

# -----------------------------------------------------------------------------
# Configuration: Load batch sizes from pyproject.toml (strict, no fallbacks)
# -----------------------------------------------------------------------------
try:
    import toml
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _CFG = toml.load(_PROJECT_ROOT / "pyproject.toml")
    _BATCH_CFG = _CFG["tool"]["flwr"]["cluster"]["batch_sizes"]
    LOGIT_BATCH_DEFAULT = int(_BATCH_CFG["logit_batch"])
    FEATURE_BATCH_DEFAULT = int(_BATCH_CFG["feature_batch"])
except Exception as e:  # pragma: no cover
    raise KeyError(
        "Missing required [tool.flwr.cluster.batch_sizes] configuration in pyproject.toml"
    ) from e

# ----------------------------------------------------------------------------
# 1.  Public reference set loader
# ----------------------------------------------------------------------------

_REF_CACHE: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

def _hash_dataset(ds) -> str:
    """Return SHA256 hash of labels to detect accidental dataset drift."""
    h = hashlib.sha256()
    # hashing only labels is enough for identity & cheap
    for _, lbl in ds:
        h.update(int(lbl).to_bytes(2, byteorder="little", signed=False))
    return h.hexdigest()


def load_reference_set(name: str = "cifar10_test", *, max_samples: int = 512) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Load CIFAR-10 reference set for clustering. Images are normalized tensors.

    The reference set is cached in the process so multiple calls are cheap.
    """
    if name in _REF_CACHE:
        imgs, lbls, digest = _REF_CACHE[name]
        return imgs, lbls, digest  # type: ignore[misc]

    if name == "cifar10_test":
        # Use CIFAR-10 test set as reference
        test_loader = get_cifar10_test_loader(batch_size=max_samples)
        batch = next(iter(test_loader))
        imgs, lbls = batch
        
        # Take only max_samples if batch is larger
        if len(imgs) > max_samples:
            imgs = imgs[:max_samples]
            lbls = lbls[:max_samples]
    else:
        raise ValueError(f"Unknown reference name: {name}, only 'cifar10_test' supported")

    # Convert labels to list for hashing
    lbl_list = lbls.tolist()
    digest = _hash_dataset([(None, l) for l in lbl_list])
    _REF_CACHE[name] = (imgs, lbls, digest)
    return imgs, lbls, digest

# ----------------------------------------------------------------------------
# 2.  Symmetric KL divergence helpers
# ----------------------------------------------------------------------------

def _safe_softmax(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).clamp_min_(eps)


def sym_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Stable symmetric KL on two 1-D probability vectors."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    kl_pq = (p * (np.log(p) - np.log(q))).sum()
    kl_qp = (q * (np.log(q) - np.log(p))).sum()
    return float(0.5 * (kl_pq + kl_qp))

# ----------------------------------------------------------------------------
# 3.  Distance matrix and clustering
# ----------------------------------------------------------------------------

def extract_logits(model: Net, imgs: torch.Tensor, batch_size: int | None = None, device: str = "cpu") -> np.ndarray:
    """Run *model* on *imgs* and return the mean logits as a NumPy vector."""
    model.eval()
    model.to(device)
    logits_accum: List[torch.Tensor] = []
    with torch.no_grad():
        bs = batch_size or LOGIT_BATCH_DEFAULT
        for i in range(0, len(imgs), bs):
            batch = imgs[i : i + bs].to(device)
            logits_accum.append(model(batch).cpu())
    logits = torch.cat(logits_accum, dim=0).mean(dim=0)  # (num_classes,)
    return logits.numpy(force=True) if hasattr(logits, "numpy") else logits.numpy()

def extract_probs(model: Net, imgs: torch.Tensor, batch_size: int | None = None, device: str = "cpu") -> np.ndarray:
    """Run *model* on *imgs* and return the mean probabilities as a NumPy vector."""
    model.eval()
    model.to(device)
    probs_accum: List[torch.Tensor] = []
    with torch.no_grad():
        bs = batch_size or LOGIT_BATCH_DEFAULT
        for i in range(0, len(imgs), bs):
            batch = imgs[i : i + bs].to(device)
            probs_accum.append(torch.softmax(model(batch).cpu(), dim=-1))
    probs = torch.cat(probs_accum, dim=0).mean(dim=0)  # (num_classes,)
    return probs.numpy(force=True) if hasattr(probs, "numpy") else probs.numpy()

def extract_features(model: Net, imgs: torch.Tensor, batch_size: int | None = None, device: str = "cpu") -> np.ndarray:
    """Extract mean penultimate-layer activations on imgs."""
    model.eval()
    model.to(device)
    feats_accum: List[torch.Tensor] = []
    with torch.no_grad():
        bs = batch_size or FEATURE_BATCH_DEFAULT
        for i in range(0, len(imgs), bs):
            batch = imgs[i : i + bs].to(device)
            # Forward through conv layers
            x = model.pool(F.relu(model.conv1(batch)))
            x = model.pool(F.relu(model.conv2(x)))
            x = x.view(x.size(0), -1)
            # Penultimate MLP
            x = F.relu(model.fc1(x))
            x = F.relu(model.fc2(x))
            feats_accum.append(x.cpu())
    feats = torch.cat(feats_accum, dim=0).mean(dim=0)
    return feats.numpy()


def distance_matrix(
    probs_list: List[np.ndarray],
    metric: str = "sym_kl",
    eps: float = 1e-6,
) -> np.ndarray:
    """Return symmetric |S|×|S| matrix for the requested metric.

    Parameters
    ----------
    metric : {"sym_kl", "js", "cosine"}
        Distance metric to use.
    """
    metric_fns = {
        "sym_kl": lambda a, b: sym_kl(a, b, eps),
        "js": lambda a, b: js_divergence(a, b, eps),
        "cosine": lambda a, b: 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)),
    }
    if metric not in metric_fns:
        raise ValueError(f"Unknown distance metric: {metric}")
    dist_fn = metric_fns[metric]

    S = len(probs_list)
    D = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(i + 1, S):
            d = dist_fn(probs_list[i], probs_list[j])
            D[i, j] = D[j, i] = d
    return D


def cluster_labels(D: np.ndarray, tau: float) -> np.ndarray:
    """Return cluster labels from a pre-computed distance matrix.

    scikit-learn ≥1.4 removed the ``affinity`` argument; use ``metric`` instead.
    """
    if len(D) == 1:
        return np.array([0])
    clustering = AgglomerativeClustering(
        metric="precomputed",  # scikit-learn ≥1.4 API
        linkage="average",
        distance_threshold=tau,
        n_clusters=None,
        compute_distances=True  # Required when using distance_threshold
    )
    return clustering.fit_predict(D)


# Remove metadata_based_clustering - not needed for dynamic weight-based clustering


def cifar10_weight_clustering(
    server_weights_list: List[List[np.ndarray]], 
    global_weights: List[np.ndarray],
    reference_imgs: torch.Tensor,
    round_num: int,
    tau: float,
    stability_history: List[dict] = None
) -> np.ndarray:
    """
    CIFAR-10 dynamic weight-based clustering using final layer similarity.
    
    Uses final layer weights for clustering instead of full gradients for efficiency.
    Better suited for CIFAR-10 classification than medical imaging approaches.
    """
    print(f"[Round {round_num}] Using dynamic weight-based clustering for CIFAR-10")
    n_servers = len(server_weights_list)
    
    # 1. Extract final layer weights for clustering
    final_layer_weights = []
    for weights in server_weights_list:
        # Take last 2 weight arrays (fc3.weight and fc3.bias for CIFAR-10 model)
        final_weights = np.concatenate([weights[-2].flatten(), weights[-1].flatten()])
        final_layer_weights.append(final_weights)
    
    # 2. Compute cosine distances between final layer weights
    weight_distances = np.zeros((n_servers, n_servers))
    for i in range(n_servers):
        for j in range(i + 1, n_servers):
            # Cosine distance
            dot_product = np.dot(final_layer_weights[i], final_layer_weights[j])
            norms = np.linalg.norm(final_layer_weights[i]) * np.linalg.norm(final_layer_weights[j])
            cosine_sim = dot_product / (norms + 1e-8)
            weight_distances[i, j] = weight_distances[j, i] = 1.0 - cosine_sim
    
    # 3. Use tau parameter from config with minimal adaptation
    print(f"[Round {round_num}] Using tau = {tau:.3f} from config")
    adaptive_tau = tau
    
    # Minimal stability adjustment for dynamic clustering
    if stability_history and len(stability_history) >= 3:
        recent_changes = sum(
            1 for i in range(-3, 0) 
            if len(set(stability_history[i].values())) > 1
        )
        if recent_changes >= 2:  # If clustering changed in 2+ recent rounds
            adaptive_tau = tau + 0.01  # Small stability boost
            print(f"[Round {round_num}] Applied stability adjustment: +0.01")
    
    print(f"[Round {round_num}] Final tau = {adaptive_tau:.3f}")
    
    return cluster_labels(weight_distances, adaptive_tau)


def extract_cifar10_features(model: Net, imgs: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Extract penultimate layer features for CIFAR-10."""
    model.eval()
    model.to(device)
    
    features_list = []
    with torch.no_grad():
        bs = FEATURE_BATCH_DEFAULT
        for i in range(0, len(imgs), bs):
            batch = imgs[i:i+bs].to(device)
            
            # Forward to penultimate layer (fc2) for CIFAR-10 model
            x = model.pool(F.relu(model.conv1(batch)))
            x = model.pool(F.relu(model.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(model.fc1(x))
            features = F.relu(model.fc2(x))  # 84-dimensional features
            
            features_list.append(features.cpu())
    
    # Return mean features across all samples
    all_features = torch.cat(features_list, dim=0)
    return all_features.mean(dim=0).numpy()


def compute_gradient_similarity(weights1: List[np.ndarray], weights2: List[np.ndarray], 
                               global_weights: List[np.ndarray]) -> float:
    """Compute cosine similarity between gradient directions."""
    grad1 = [w1 - gw for w1, gw in zip(weights1, global_weights)]
    grad2 = [w2 - gw for w2, gw in zip(weights2, global_weights)]
    
    # Flatten gradients
    grad1_flat = np.concatenate([g.flatten() for g in grad1])
    grad2_flat = np.concatenate([g.flatten() for g in grad2])
    
    # Cosine similarity
    dot_product = np.dot(grad1_flat, grad2_flat)
    norms = np.linalg.norm(grad1_flat) * np.linalg.norm(grad2_flat)
    
    return dot_product / (norms + 1e-8)


# ----------------------------------------------------------------------------
# 4.  Rebuild model from raw NumPy weight list (helper for cloud server)
# ----------------------------------------------------------------------------

def rebuild_model_from_weights(weights: List[np.ndarray]) -> Net:
    model = Net()
    set_weights(model, weights)
    return model
