"""
Fedge-Simulation: Hierarchical FL with Clustering - In-Memory Simulation Mode for CIFAR-10

This orchestrator runs the entire Fedge pipeline in a single process with shared memory,
following the HHAR Fedge-Simulation architecture for memory efficiency and speed.

Architecture:
- 3-level hierarchy: clients -> leaf servers -> cloud
- Dynamic clustering at cloud level using cosine similarity
- FedProx optimization (SCAFFOLD disabled by default)
- Accuracy gate for parameter acceptance

Usage:
    cd Fedge-Simulation
    SEED=42 python orchestrator.py
"""

from __future__ import annotations
import os
import sys
import json
import time
import random
import logging
import gc
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import toml

# Add fedge module to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedge.task import (
    Net, load_cifar10_data, train, test, get_weights, set_weights, Cifar10Data
)
from fedge.cluster_utils import cifar10_weight_clustering as weight_clustering
from fedge.cluster_utils import gradient_based_clustering
from fedge.partitioning import hier_dirichlet_indices, write_partitions
from fedge.stats import _mean_std_ci
from fedge.server_scaffold import ServerSCAFFOLD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration Loading
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = toml.load(PROJECT_ROOT / "pyproject.toml")

# Hierarchy config
HIER_CFG = CONFIG["tool"]["flwr"]["hierarchy"]
NUM_SERVERS = int(HIER_CFG["num_servers"])
CLIENTS_PER_SERVER = list(HIER_CFG["clients_per_server"])
GLOBAL_ROUNDS = int(HIER_CFG["global_rounds"])
SERVER_ROUNDS_PER_GLOBAL = int(HIER_CFG["server_rounds_per_global"])

# Learning parameters
LR_INIT = float(HIER_CFG["lr_init"])
SERVER_LR = float(HIER_CFG.get("server_lr", 1.0))
GLOBAL_LR = float(HIER_CFG.get("global_lr", 1.0))
LOCAL_EPOCHS = int(HIER_CFG["local_epochs"])
BATCH_SIZE = 64  # Matches FedProx/HierFL for fair comparison
EVAL_BATCH_SIZE = int(HIER_CFG.get("eval_batch_size", 32))

# Optimization parameters
WEIGHT_DECAY = float(HIER_CFG["weight_decay"])
CLIP_NORM = float(HIER_CFG["clip_norm"])
MOMENTUM = float(HIER_CFG["momentum"])
LR_GAMMA = float(HIER_CFG["lr_gamma"])
PROX_MU = float(HIER_CFG["prox_mu"])
SCAFFOLD_ENABLED = bool(HIER_CFG.get("scaffold_enabled", False))

# Clustering config
CLUSTER_CFG = CONFIG["tool"]["flwr"]["cloud_cluster"]
CLUSTER_ENABLED = bool(CLUSTER_CFG["enable"])
CLUSTER_START_ROUND = int(CLUSTER_CFG["start_round"])
CLUSTER_FREQUENCY = int(CLUSTER_CFG["frequency"])
CLUSTER_TAU = float(CLUSTER_CFG["tau"])
CLUSTER_METHOD = str(CLUSTER_CFG.get("method", "weight"))  # "weight" or "gradient"

# Accuracy gate
CLUSTER_BETTER_DELTA = float(HIER_CFG.get("cluster_better_delta", 0.0))

# SCAFFOLD config (v9: fixed amplification and clipping bugs)
SCAFFOLD_SCALING_FACTOR = float(HIER_CFG.get("scaffold_scaling_factor", 0.1))   # v9: replaces 1/(K*lr)
SCAFFOLD_CORRECTION_CLIP = float(HIER_CFG.get("scaffold_correction_clip", 0.1)) # v9: clip corrections
SCAFFOLD_WARMUP_ROUNDS = int(HIER_CFG.get("scaffold_warmup_rounds", 10))        # v9: gradual activation
SCAFFOLD_CLIP_VALUE = float(HIER_CFG.get("scaffold_clip_value", 1.0))           # v9: tightened to 1.0

# Server-level SCAFFOLD config (disabled by default)
SCAFFOLD_SERVER_ENABLED = bool(HIER_CFG.get("scaffold_server_enabled", False))
SCAFFOLD_SERVER_LR = float(HIER_CFG.get("scaffold_server_lr", 1.0))
SCAFFOLD_CORRECTION_LR = float(HIER_CFG.get("scaffold_correction_lr", 0.1))

# Server isolation config (v6)
SERVER_ISOLATION = bool(HIER_CFG.get("server_isolation", False))

# Seed
SEED = int(os.environ.get("SEED", HIER_CFG.get("dirichlet", {}).get("seed", 42)))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for faster convolutions

logger.info(f"[CONFIG] SEED={SEED}, NUM_SERVERS={NUM_SERVERS}, CLIENTS_PER_SERVER={CLIENTS_PER_SERVER}")
logger.info(f"[CONFIG] GLOBAL_ROUNDS={GLOBAL_ROUNDS}, LOCAL_EPOCHS={LOCAL_EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
logger.info(f"[CONFIG] SCAFFOLD={SCAFFOLD_ENABLED}, PROX_MU={PROX_MU}, CLUSTER_TAU={CLUSTER_TAU}")

# ==============================================================================
# Data Loading and Partitioning
# ==============================================================================

def create_partitions(train_labels: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
    """Create Dirichlet-based partitions for hierarchical FL.

    Returns:
        Dict mapping server_id -> client_id -> list of train sample indices
    """
    # Check for existing partition file
    parts_json = PROJECT_ROOT / "rounds" / f"partitions_seed{SEED}.json"
    if parts_json.exists():
        logger.info(f"Loading existing partitions from {parts_json}")
        with open(parts_json) as f:
            return json.load(f)

    logger.info("Creating new Dirichlet partitions...")

    # Create hierarchical Dirichlet partition
    partitions = hier_dirichlet_indices(
        labels=train_labels,
        num_servers=NUM_SERVERS,
        clients_per_server=CLIENTS_PER_SERVER,
        seed=SEED
    )

    # Log partition sizes
    for sid in range(NUM_SERVERS):
        for cid in range(CLIENTS_PER_SERVER[sid]):
            n_samples = len(partitions[str(sid)][str(cid)])
            logger.info(f"Server {sid}, Client {cid}: {n_samples} samples")

    # Save partitions
    parts_json.parent.mkdir(parents=True, exist_ok=True)
    write_partitions(parts_json, partitions)
    logger.info(f"Saved partitions to {parts_json}")

    return partitions


# ==============================================================================
# Simulation Components
# ==============================================================================

class SimulatedClient:
    """In-memory client for simulation mode."""

    def __init__(
        self,
        client_id: str,
        server_id: int,
        train_indices: List[int],
        test_indices: List[int],
        train_dataset,
        test_dataset,
        device: torch.device
    ):
        self.client_id = client_id
        self.server_id = server_id
        self.device = device

        # Create data loaders using subset of shared datasets
        from torch.utils.data import Subset, DataLoader

        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

        self.trainloader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True,
            pin_memory=torch.cuda.is_available(), num_workers=0
        )
        self.testloader = DataLoader(
            test_subset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
            pin_memory=torch.cuda.is_available(), num_workers=0
        )

        self.num_train = len(train_indices)
        self.num_test = len(test_indices)

        # SCAFFOLD state (disabled by default)
        self.scaffold_manager = None
        if SCAFFOLD_ENABLED:
            from fedge.scaffold_utils import create_scaffold_manager
            net = Net().to(self.device)
            self.scaffold_manager = create_scaffold_manager(net)
            for name in self.scaffold_manager.client_control:
                self.scaffold_manager.client_control[name] = self.scaffold_manager.client_control[name].to(self.device)
            for name in self.scaffold_manager.server_control:
                self.scaffold_manager.server_control[name] = self.scaffold_manager.server_control[name].to(self.device)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
        ref_weights: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train locally and return updated weights."""
        # Create local model for training
        net = Net()
        set_weights(net, parameters)
        net.to(self.device)

        # Only create global_net if SCAFFOLD is enabled (expensive to keep both)
        global_net = None
        if SCAFFOLD_ENABLED:
            global_net = Net()
            set_weights(global_net, parameters)
            global_net.to(self.device)

        # Apply SCAFFOLD if enabled
        if SCAFFOLD_ENABLED and self.scaffold_manager:
            net._scaffold_manager = self.scaffold_manager
            if "server_control" in config and config["server_control"] is not None:
                server_control = config["server_control"]
                for name in server_control:
                    server_control[name] = server_control[name].to(self.device)
                self.scaffold_manager.server_control = server_control
            for name in self.scaffold_manager.client_control:
                self.scaffold_manager.client_control[name] = self.scaffold_manager.client_control[name].to(self.device)

        lr = config.get("lr", LR_INIT)
        epochs = config.get("epochs", LOCAL_EPOCHS)
        global_round = config.get("global_round", 0)

        # v9: SCAFFOLD is always active if enabled, warmup is handled internally
        # via warmup_factor in apply_scaffold_correction
        scaffold_active = SCAFFOLD_ENABLED

        # Train
        loss = train(
            net=net,
            loader=self.trainloader,
            epochs=epochs,
            device=self.device,
            lr=lr,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            clip_norm=CLIP_NORM,
            prox_mu=config.get("prox_mu", PROX_MU),
            ref_weights=ref_weights,
            global_round=global_round,
            scaffold_enabled=scaffold_active,
            scaffold_warmup_rounds=SCAFFOLD_WARMUP_ROUNDS,    # v9: passed to train
            scaffold_correction_clip=SCAFFOLD_CORRECTION_CLIP # v9: passed to train
        )

        # Evaluate
        eval_loss, accuracy = test(net, self.testloader, self.device)

        # Get updated weights
        new_weights = get_weights(net)

        # SCAFFOLD: update client control variate
        scaffold_delta = None
        if scaffold_active and self.scaffold_manager:
            from fedge.scaffold_utils import aggregate_scaffold_controls
            self.scaffold_manager.update_client_control(
                local_model=net,
                global_model=global_net,
                learning_rate=lr,
                local_epochs=epochs,
                clip_value=SCAFFOLD_CLIP_VALUE,           # v9: tightened to 1.0
                scaling_factor=SCAFFOLD_SCALING_FACTOR    # v9: replaces 1/(K*lr) division
            )
            scaffold_delta = self.scaffold_manager.get_client_control()

        metrics = {
            "train_loss": loss,
            "eval_loss": eval_loss,
            "accuracy": accuracy,
            "num_examples": self.num_train,
            "scaffold_delta": scaffold_delta
        }

        return new_weights, self.num_train, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate on local test data."""
        net = Net()
        set_weights(net, parameters)
        net.to(self.device)

        loss, accuracy = test(net, self.testloader, self.device)

        return loss, self.num_test, {"accuracy": accuracy}


class SimulatedLeafServer:
    """In-memory leaf server for simulation mode."""

    def __init__(
        self,
        server_id: int,
        clients: List[SimulatedClient],
        device: torch.device
    ):
        self.server_id = server_id
        self.clients = clients
        self.device = device

        # Server state
        self.latest_parameters: Optional[List[np.ndarray]] = None
        self.c_global: Optional[Dict[str, torch.Tensor]] = None  # SCAFFOLD server control
        self.prev_accuracy = 0.0

        # Metrics
        self.round_metrics = []

    def run_round(
        self,
        global_round: int,
        initial_parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], Dict]:
        """Run one round of local FL with all clients."""

        # Apply accuracy gate: compare new vs old parameters
        if self.latest_parameters is not None and CLUSTER_BETTER_DELTA > 0:
            net = Net()
            set_weights(net, initial_parameters)
            net.to(self.device)

            total_loss, total_acc, total_samples = 0.0, 0.0, 0
            for client in self.clients:
                loss, num, metrics = client.evaluate(initial_parameters, config)
                total_loss += loss * num
                total_acc += metrics["accuracy"] * num
                total_samples += num

            new_acc = total_acc / total_samples if total_samples > 0 else 0.0

            if new_acc < self.prev_accuracy - CLUSTER_BETTER_DELTA:
                logger.info(f"[Server {self.server_id}] Accuracy gate REJECTED: {new_acc:.4f} < {self.prev_accuracy:.4f}")
                initial_parameters = self.latest_parameters
            else:
                self.prev_accuracy = new_acc

        # Prepare config with SCAFFOLD control variates
        client_config = {
            "epochs": LOCAL_EPOCHS,
            "lr": config.get("lr", LR_INIT),
            "prox_mu": config.get("prox_mu", PROX_MU),
            "global_round": global_round,
            "server_control": self.c_global
        }

        # Collect client updates
        client_results = []
        for client in self.clients:
            weights, num_examples, metrics = client.fit(
                parameters=initial_parameters,
                config=client_config,
                ref_weights=initial_parameters
            )
            client_results.append((weights, num_examples, metrics))

        # Aggregate client weights (FedAvg)
        total_examples = sum(num for _, num, _ in client_results)
        aggregated_weights = []

        for layer_idx in range(len(initial_parameters)):
            weighted_sum = np.zeros_like(initial_parameters[layer_idx])
            for weights, num_examples, _ in client_results:
                weighted_sum += weights[layer_idx] * num_examples
            aggregated_weights.append(weighted_sum / total_examples)

        # SCAFFOLD: aggregate control variates
        if SCAFFOLD_ENABLED:
            from fedge.scaffold_utils import aggregate_scaffold_controls
            scaffold_deltas = [m["scaffold_delta"] for _, _, m in client_results if m.get("scaffold_delta")]
            if scaffold_deltas:
                client_weights = [float(num) / total_examples for _, num, _ in client_results]
                self.c_global = aggregate_scaffold_controls(scaffold_deltas, client_weights)

        # Update latest parameters
        self.latest_parameters = aggregated_weights

        # SERVER-LEVEL EVALUATION
        server_net = Net()
        set_weights(server_net, aggregated_weights)
        server_net.to(self.device)

        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for client in self.clients:
                for batch in client.testloader:
                    images, labels = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)
                    outputs = server_net(images)
                    loss = torch.nn.functional.cross_entropy(outputs, labels, reduction='sum')
                    total_test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_test_correct += (predicted == labels).sum().item()
                    total_test_samples += labels.size(0)

        server_test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0.0
        server_test_accuracy = total_test_correct / total_test_samples if total_test_samples > 0 else 0.0

        server_metrics = {
            "server_id": self.server_id,
            "global_round": global_round,
            "server_test_loss": server_test_loss,
            "server_test_accuracy": server_test_accuracy,
            "num_clients": len(self.clients),
            "total_train_samples": total_examples,
            "total_test_samples": total_test_samples
        }

        self.round_metrics.append(server_metrics)
        logger.info(f"[Server {self.server_id}] Round {global_round}: test_loss={server_test_loss:.4f}, test_acc={server_test_accuracy:.4f}")

        return aggregated_weights, server_metrics


class CloudAggregator:
    """Cloud-level aggregation with dynamic clustering."""

    def __init__(self, num_servers: int, device: torch.device, clusters_dir: Path):
        self.num_servers = num_servers
        self.device = device
        self.clusters_dir = clusters_dir

        # Clustering state
        self.cluster_map: Dict[int, int] = {}
        self.cluster_parameters: Dict[int, List[np.ndarray]] = {}

        # Track previous server weights for gradient-based clustering
        self.previous_server_weights: Dict[int, List[np.ndarray]] = {}

        # Server-level SCAFFOLD (v6)
        self.server_scaffold = ServerSCAFFOLD(num_servers=num_servers) if SCAFFOLD_SERVER_ENABLED else None

        # Metrics
        self.round_metrics = []
        self.cluster_history = []

    def aggregate_and_cluster(
        self,
        server_models: List[Tuple[int, List[np.ndarray], int]],
        global_round: int
    ) -> Dict[int, List[np.ndarray]]:
        """Aggregate server models and perform clustering."""

        server_ids = [sid for sid, _, _ in server_models]
        weights_list = [w for _, w, _ in server_models]
        sample_counts = [n for _, _, n in server_models]
        total_samples = sum(sample_counts)

        # Step 1: Global aggregation (FedAvg across all servers)
        global_weights = []
        for layer_idx in range(len(weights_list[0])):
            weighted_sum = np.zeros_like(weights_list[0][layer_idx])
            for weights, num_samples in zip(weights_list, sample_counts):
                weighted_sum += weights[layer_idx] * num_samples
            global_weights.append(weighted_sum / total_samples)

        # Step 2: Clustering (if enabled and conditions met)
        should_cluster = (
            CLUSTER_ENABLED and
            global_round >= CLUSTER_START_ROUND and
            (global_round - CLUSTER_START_ROUND) % CLUSTER_FREQUENCY == 0
        )

        if should_cluster:
            logger.info(f"[Cloud] Running clustering at round {global_round} (method={CLUSTER_METHOD})")

            if CLUSTER_METHOD == "gradient":
                # Gradient-based clustering: use direction of model updates
                # previous_server_weights is guaranteed to have data by round 30
                # (populated every round since round 1)
                previous_weights_list = [
                    self.previous_server_weights[sid] for sid in server_ids
                ]
                labels, similarity_matrix, tau = gradient_based_clustering(
                    server_weights_list=weights_list,
                    previous_weights_list=previous_weights_list,
                    tau=CLUSTER_TAU,
                    round_num=global_round
                )
            else:
                # Weight-based clustering
                labels, similarity_matrix, tau = weight_clustering(
                    server_weights_list=weights_list,
                    global_weights=global_weights,
                    reference_imgs=None,
                    round_num=global_round,
                    tau=CLUSTER_TAU
                )

            self.cluster_map = {sid: int(labels[i]) for i, sid in enumerate(server_ids)}

            unique_clusters = np.unique(labels)
            self.cluster_parameters = {}

            for cluster_id in unique_clusters:
                cluster_mask = labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                cluster_samples = sum(sample_counts[i] for i in cluster_indices)

                cluster_weights = []
                for layer_idx in range(len(weights_list[0])):
                    weighted_sum = np.zeros_like(weights_list[0][layer_idx])
                    for i in cluster_indices:
                        weighted_sum += weights_list[i][layer_idx] * sample_counts[i]
                    cluster_weights.append(weighted_sum / cluster_samples)

                self.cluster_parameters[int(cluster_id)] = cluster_weights

            self._save_cluster_artifacts(global_round, labels, similarity_matrix, server_ids)
            logger.info(f"[Cloud] Clustering result: {len(unique_clusters)} clusters, map={self.cluster_map}")
        else:
            # Before clustering starts
            if SERVER_ISOLATION:
                # v6: Each server keeps its own model (allows divergence)
                self.cluster_map = {sid: sid for sid in server_ids}
                self.cluster_parameters = {
                    sid: weights_list[i] for i, sid in enumerate(server_ids)
                }
                logger.debug(f"[Cloud] Server isolation: each server keeps own model")
            else:
                # v1-v5: All servers get the global model
                self.cluster_map = {sid: 0 for sid in server_ids}
                self.cluster_parameters = {0: global_weights}

        # Store current weights for next round's gradient computation
        for i, sid in enumerate(server_ids):
            self.previous_server_weights[sid] = weights_list[i]

        # Server-level SCAFFOLD: update control variates (v6)
        if self.server_scaffold is not None and global_round >= SCAFFOLD_WARMUP_ROUNDS:
            for i, sid in enumerate(server_ids):
                cluster_id = self.cluster_map[sid]
                theta_cluster = self.cluster_parameters[cluster_id]

                self.server_scaffold.update_server_control(
                    server_id=sid,
                    theta_server=weights_list[i],
                    theta_cluster=theta_cluster,
                    n_samples=sample_counts[i],
                    K=SERVER_ROUNDS_PER_GLOBAL,
                    eta=SCAFFOLD_SERVER_LR,
                    clip_value=SCAFFOLD_CLIP_VALUE
                )

            self.server_scaffold.update_global_control()

            # Log server divergences
            divergences = self.server_scaffold.get_server_divergence()
            logger.info(f"[Cloud] Server-level SCAFFOLD divergences: {divergences}")

        metrics = {
            "global_round": global_round,
            "num_clusters": len(self.cluster_parameters),
            "cluster_map": self.cluster_map.copy()
        }
        self.round_metrics.append(metrics)

        return self.cluster_parameters

    def get_cluster_model(self, server_id: int, apply_scaffold: bool = True) -> List[np.ndarray]:
        """Get the cluster-specific model for a server, with optional SCAFFOLD correction.

        Args:
            server_id: Server ID
            apply_scaffold: Whether to apply server-level SCAFFOLD correction

        Returns:
            Model weights (possibly corrected)
        """
        cluster_id = self.cluster_map.get(server_id, 0)
        theta_cluster = self.cluster_parameters.get(cluster_id, list(self.cluster_parameters.values())[0])

        # Apply server-level SCAFFOLD correction (v6)
        if apply_scaffold and self.server_scaffold is not None:
            return self.server_scaffold.apply_correction(
                server_id=server_id,
                theta_cluster=theta_cluster,
                correction_lr=SCAFFOLD_CORRECTION_LR
            )

        return theta_cluster

    def _save_cluster_artifacts(
        self,
        global_round: int,
        labels: np.ndarray,
        similarity_matrix: np.ndarray,
        server_ids: List[int]
    ):
        """Save clustering artifacts to clusters subfolder."""
        import pandas as pd

        assignments = {str(sid): int(labels[i]) for i, sid in enumerate(server_ids)}
        with open(self.clusters_dir / f"assignments_g{global_round}.json", "w") as f:
            json.dump(assignments, f, indent=2)

        sim_df = pd.DataFrame(
            similarity_matrix,
            index=[f"s{i}" for i in server_ids],
            columns=[f"s{i}" for i in server_ids]
        )
        sim_df.to_csv(self.clusters_dir / f"similarity_g{global_round}.csv")

        self.cluster_history.append({
            "round": global_round,
            "labels": labels.tolist(),
            "cluster_map": assignments
        })


# ==============================================================================
# Main Orchestrator
# ==============================================================================

class SimulationOrchestrator:
    """Main orchestrator for in-memory hierarchical FL simulation."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Orchestrator] Using device: {self.device}")

        # Load data ONCE (following HHAR Fedge-Simulation pattern)
        logger.info("[Orchestrator] Loading CIFAR-10 dataset (one-time)...")
        self.cifar_data: Cifar10Data = load_cifar10_data(seed=SEED)
        logger.info(f"[Orchestrator] Dataset loaded: {len(self.cifar_data.train)} train, "
                   f"{len(self.cifar_data.test)} test samples")

        # Create partitions for train data
        self.partitions = create_partitions(self.cifar_data.train_labels)

        # Initialize model weights (used only for first round initialization)
        # NOTE: No global model maintained - only cluster-specific models
        init_model = Net()
        self.initial_weights = get_weights(init_model)

        # Metrics collection - CLEAN CONSOLIDATED STRUCTURE
        # Create directories BEFORE initializing components (cloud needs clusters_dir)
        self.run_dir = PROJECT_ROOT / "runs" / f"seed{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # All metrics in one place (server-level aggregated, no global model)
        self.global_csv = self.run_dir / "server_rounds.csv"

        # Server metrics in subfolder
        self.servers_dir = self.run_dir / "servers"
        self.servers_dir.mkdir(parents=True, exist_ok=True)
        self.server_csvs = {
            sid: self.servers_dir / f"server_{sid}.csv"
            for sid in range(NUM_SERVERS)
        }

        # Cluster artifacts in subfolder
        self.clusters_dir = self.run_dir / "clusters"
        self.clusters_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []

        # Initialize components (servers, clients, cloud)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize leaf servers and clients."""
        logger.info("[Orchestrator] Initializing leaf servers and clients...")

        self.leaf_servers: List[SimulatedLeafServer] = []

        # Calculate test indices for each client (proportional to train)
        total_test = len(self.cifar_data.test)
        total_train = len(self.cifar_data.train)

        for server_id in range(NUM_SERVERS):
            clients = []

            for client_id in range(CLIENTS_PER_SERVER[server_id]):
                # Get train partition indices
                train_indices = self.partitions[str(server_id)][str(client_id)]

                # Create proportional test indices
                # Each client gets test samples proportional to their train share
                n_test = max(1, int(len(train_indices) / total_train * total_test))

                # Deterministic test assignment based on client ID
                np.random.seed(SEED + server_id * 100 + client_id)
                test_indices = np.random.choice(total_test, size=n_test, replace=False).tolist()

                client = SimulatedClient(
                    client_id=f"{server_id}_{client_id}",
                    server_id=server_id,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_dataset=self.cifar_data.train,
                    test_dataset=self.cifar_data.test,
                    device=self.device
                )
                clients.append(client)

            server = SimulatedLeafServer(
                server_id=server_id,
                clients=clients,
                device=self.device
            )
            self.leaf_servers.append(server)

            total_train_samples = sum(c.num_train for c in clients)
            total_test_samples = sum(c.num_test for c in clients)
            logger.info(f"[Orchestrator] Server {server_id}: {len(clients)} clients, "
                       f"{total_train_samples} train / {total_test_samples} test samples")

        # Initialize cloud aggregator
        self.cloud = CloudAggregator(NUM_SERVERS, self.device, self.clusters_dir)

    def run_global_round(self, global_round: int) -> Dict:
        """Run one global round of hierarchical FL."""
        round_start = time.time()

        server_models = []

        for server in self.leaf_servers:
            if self.cloud.cluster_parameters:
                # Use cluster-specific model for this server
                initial_params = self.cloud.get_cluster_model(server.server_id)
            else:
                # First round only: use initial random weights
                initial_params = self.initial_weights

            config = {
                "lr": LR_INIT * (LR_GAMMA ** (global_round - 1)),
                "prox_mu": PROX_MU
            }

            aggregated_weights, metrics = server.run_round(
                global_round=global_round,
                initial_parameters=initial_params,
                config=config
            )

            total_samples = sum(c.num_train for c in server.clients)
            server_models.append((server.server_id, aggregated_weights, total_samples))

        # Cloud aggregation + clustering (no global model - only cluster models)
        self.cloud.aggregate_and_cluster(server_models, global_round)

        # NOTE: Global weights update REMOVED per design:
        # Fedge uses cluster-specific models, not a single global model.
        # Each server receives its cluster's aggregated model.

        # Compute round metrics with t-critical based 95% CI
        round_time = time.time() - round_start
        server_accuracies = [s.round_metrics[-1]["server_test_accuracy"] for s in self.leaf_servers]
        server_losses = [s.round_metrics[-1]["server_test_loss"] for s in self.leaf_servers]

        # Use proper t-distribution CI calculation (consistent with HHAR)
        acc_mean, acc_std, acc_ci_low, acc_ci_high = _mean_std_ci(server_accuracies)
        loss_mean, loss_std, loss_ci_low, loss_ci_high = _mean_std_ci(server_losses)

        metrics = {
            "global_round": global_round,
            "avg_accuracy": acc_mean,
            "avg_loss": loss_mean,
            "accuracy_std": acc_std,
            "accuracy_ci_low": acc_ci_low,
            "accuracy_ci_high": acc_ci_high,
            "accuracy_min": min(server_accuracies),
            "accuracy_max": max(server_accuracies),
            "loss_std": loss_std,
            "loss_ci_low": loss_ci_low,
            "loss_ci_high": loss_ci_high,
            "num_clusters": len(self.cloud.cluster_parameters),
            "cluster_map": self.cloud.cluster_map.copy(),
            "round_time": round_time
        }

        self.metrics_history.append(metrics)

        # Save metrics immediately
        self._save_global_round_metrics(metrics)
        for server in self.leaf_servers:
            self._save_server_round_metrics(server.server_id, server.round_metrics[-1])

        logger.info(f"[Round {global_round}] acc={acc_mean:.4f} (CI: {acc_ci_low:.4f}-{acc_ci_high:.4f}), "
                   f"loss={loss_mean:.4f}, clusters={len(self.cloud.cluster_parameters)}, time={round_time:.1f}s")

        return metrics

    def run(self):
        """Run the full simulation."""
        logger.info(f"[Orchestrator] Starting simulation: {GLOBAL_ROUNDS} global rounds")
        start_time = time.time()

        for global_round in range(1, GLOBAL_ROUNDS + 1):
            self.run_global_round(global_round)

            # Periodic garbage collection (HARDCODED like HHAR - every 10 rounds)
            if global_round % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total_time = time.time() - start_time

        self._save_cluster_history()

        logger.info(f"[Orchestrator] Simulation complete!")
        logger.info(f"[Orchestrator] Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        logger.info(f"[Orchestrator] Average time per round: {total_time/GLOBAL_ROUNDS:.1f}s")

        final_acc = self.metrics_history[-1]["avg_accuracy"]
        logger.info(f"[Orchestrator] Final accuracy: {final_acc:.4f}")

    def _save_global_round_metrics(self, metrics: Dict):
        """Append one row to global_rounds.csv with t-critical based 95% CI."""
        fieldnames = [
            "global_round", "avg_accuracy", "avg_loss",
            "accuracy_std", "accuracy_ci_low", "accuracy_ci_high",
            "accuracy_min", "accuracy_max",
            "loss_std", "loss_ci_low", "loss_ci_high",
            "num_clusters", "cluster_map", "round_time"
        ]

        write_header = not self.global_csv.exists()
        with open(self.global_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            writer.writerow({
                "global_round": metrics["global_round"],
                "avg_accuracy": metrics["avg_accuracy"],
                "avg_loss": metrics["avg_loss"],
                "accuracy_std": metrics["accuracy_std"],
                "accuracy_ci_low": metrics["accuracy_ci_low"],
                "accuracy_ci_high": metrics["accuracy_ci_high"],
                "accuracy_min": metrics["accuracy_min"],
                "accuracy_max": metrics["accuracy_max"],
                "loss_std": metrics["loss_std"],
                "loss_ci_low": metrics["loss_ci_low"],
                "loss_ci_high": metrics["loss_ci_high"],
                "num_clusters": metrics["num_clusters"],
                "cluster_map": json.dumps(metrics["cluster_map"]),
                "round_time": metrics["round_time"]
            })

    def _save_server_round_metrics(self, server_id: int, metrics: Dict):
        """Append one row to server_{id}.csv (simplified schema)."""
        fieldnames = [
            "server_id", "global_round",
            "server_test_loss", "server_test_accuracy",
            "num_clients", "total_train_samples", "total_test_samples"
        ]

        csv_path = self.server_csvs[server_id]
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            writer.writerow(metrics)

    def _save_cluster_history(self):
        """Save cluster history to JSON in clusters subfolder."""
        if self.cloud.cluster_history:
            with open(self.clusters_dir / "cluster_history.json", "w") as f:
                json.dump(self.cloud.cluster_history, f, indent=2)
        logger.info(f"[Orchestrator] All metrics saved to {self.run_dir}")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Fedge-Simulation: Hierarchical FL with Clustering (CIFAR-10)")
    logger.info("=" * 60)

    orchestrator = SimulationOrchestrator()
    orchestrator.run()
