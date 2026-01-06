"""Hierarchical Dirichlet partition utilities.

This module produces a two-level Dirichlet split compatible with the
HierFL paper/codebase:
    1.  Outer Dirichlet across *servers* (alpha_server)
    2.  Inner Dirichlet across *clients within each server* (alpha_client)

The result is stored as a JSON mapping
    {
        "0": {"0": [..indices..], "1": [...], ...},
        "1": {...},
        ...
    }
where top-level keys are server_ids (str), second-level keys are
client_ids (str) and values are *global* indices into the original train
set.  Indices are plain Python ints so the file is portable and does not
require NumPy when loading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

__all__ = [
    "hier_dirichlet_indices",
    "write_partitions",
]


def _dirichlet_partition_indices(
    labels: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int,
    min_partition_size: int = 10,
) -> Dict[int, List[int]]:
    """Partition indices using Dirichlet distribution over label proportions.

    This is a direct implementation that doesn't rely on private APIs.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer labels for each sample.
    num_partitions : int
        Number of partitions to create.
    alpha : float
        Dirichlet concentration parameter. Lower values create more heterogeneous
        partitions. alpha < 1 = highly non-IID, alpha = 1 = uniform, alpha > 1 = IID-ish.
    seed : int
        Random seed for reproducibility.
    min_partition_size : int
        Minimum number of samples per partition.

    Returns
    -------
    Dict[int, List[int]]
        Mapping from partition_id to list of global indices.
    """
    rng = np.random.default_rng(seed)

    # Get unique labels and their indices
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Group indices by label
    label_to_indices: Dict[int, List[int]] = {
        int(label): np.where(labels == label)[0].tolist()
        for label in unique_labels
    }

    # Shuffle indices within each label group
    for label in label_to_indices:
        rng.shuffle(label_to_indices[label])

    # Sample Dirichlet proportions for each partition
    # Shape: (num_partitions, num_classes)
    proportions = rng.dirichlet([alpha] * num_classes, size=num_partitions)

    # Initialize partition assignments
    partition_indices: Dict[int, List[int]] = {i: [] for i in range(num_partitions)}

    # Assign samples from each class to partitions according to Dirichlet proportions
    for class_idx, label in enumerate(unique_labels):
        class_indices = label_to_indices[int(label)]
        num_samples = len(class_indices)

        # Calculate how many samples each partition gets from this class
        class_proportions = proportions[:, class_idx]
        class_proportions = class_proportions / class_proportions.sum()  # Normalize

        # Allocate samples
        allocated = 0
        for pid in range(num_partitions):
            if pid == num_partitions - 1:
                # Last partition gets remaining samples
                count = num_samples - allocated
            else:
                count = int(num_samples * class_proportions[pid])

            start_idx = allocated
            end_idx = min(allocated + count, num_samples)
            partition_indices[pid].extend(class_indices[start_idx:end_idx])
            allocated = end_idx

    # Shuffle indices within each partition
    for pid in partition_indices:
        rng.shuffle(partition_indices[pid])

    # Ensure minimum partition size by redistributing if needed
    total_samples = sum(len(v) for v in partition_indices.values())
    if total_samples > 0 and min_partition_size > 0:
        for pid in range(num_partitions):
            if len(partition_indices[pid]) < min_partition_size:
                # Try to borrow from larger partitions
                for donor_pid in range(num_partitions):
                    if donor_pid != pid and len(partition_indices[donor_pid]) > min_partition_size + 10:
                        needed = min_partition_size - len(partition_indices[pid])
                        available = len(partition_indices[donor_pid]) - min_partition_size
                        transfer = min(needed, available)
                        if transfer > 0:
                            partition_indices[pid].extend(partition_indices[donor_pid][-transfer:])
                            partition_indices[donor_pid] = partition_indices[donor_pid][:-transfer]
                        if len(partition_indices[pid]) >= min_partition_size:
                            break

    return partition_indices


def hier_dirichlet_indices(
    labels: np.ndarray,
    num_servers: int,
    clients_per_server: Union[int, Sequence[int]],
    *,
    alpha_server: float = 0.5,
    alpha_client: float = 0.3,
    seed: int = 42,
) -> Dict[str, Dict[str, List[int]]]:
    """Return a hierarchical mapping of indices for *all* clients.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer labels for each sample.
    num_servers : int
        Number of leaf servers.
    clients_per_server : int | Sequence[int]
        Number of clients *inside each* leaf server.
    alpha_server/alpha_client : float
        Dirichlet concentration parameters for outer / inner split.
    seed : int
        Random seed for reproducibility.
    """
    # Standardise clients_per_server into a list per server
    if isinstance(clients_per_server, int):
        cps_list: list[int] = [clients_per_server] * num_servers
    else:
        cps_list = list(clients_per_server)
        if len(cps_list) != num_servers:
            raise ValueError(f"len(clients_per_server)={len(cps_list)} does not match num_servers={num_servers}")

    # ── Outer split: across servers ────────────────────────────────────────
    server_partition = _dirichlet_partition_indices(
        labels=labels,
        num_partitions=num_servers,
        alpha=alpha_server,
        seed=seed,
        min_partition_size=50,
    )

    mapping: Dict[str, Dict[str, List[int]]] = {}

    # ── Inner split: per-server across that server's clients ───────────────
    for sid in range(num_servers):
        server_indices = server_partition[sid]
        server_labels = labels[server_indices]

        client_partition = _dirichlet_partition_indices(
            labels=server_labels,
            num_partitions=cps_list[sid],
            alpha=alpha_client,
            seed=seed + sid,  # Different seed per server for variety
            min_partition_size=10,
        )

        client_map: Dict[str, List[int]] = {}
        for cid in range(cps_list[sid]):
            # Indices from client_partition are relative to server_indices
            rel_idx = client_partition[cid]
            # Translate back to global indices
            global_idx = [int(server_indices[i]) for i in rel_idx]
            client_map[str(cid)] = global_idx
        mapping[str(sid)] = client_map

    return mapping

def write_partitions(path: Path | str, mapping: Dict[str, Dict[str, List[int]]]) -> None:
    """Write JSON mapping to *path* (overwrites if exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(mapping, fp)
