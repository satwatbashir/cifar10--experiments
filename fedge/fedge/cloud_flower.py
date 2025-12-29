# cloud_flower.py
"""
Cloud server script for hierarchical federated learning using Flower ≥1.13.
This script starts a Flower ServerApp with communication-cost modifiers,
writes communication metrics to CSV after completion, and handles
dynamic clustering and signal files.
"""

# Suppress warnings
import warnings
import sys
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="flwr")
warnings.simplefilter("ignore")

# Set environment variable to suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

import time
import json
import pickle
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import toml
import numpy as np

from flwr.server import start_server, ServerConfig
from flwr.common import Metrics, Parameters, NDArrays
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from fedge.task import Net, get_weights, set_weights, test, get_cifar10_test_loader, load_cifar10_hf, subset, make_loader
from fedge.cluster_utils import cifar10_weight_clustering
# from fedge.utils.cloud_metrics import get_cloud_metrics_collector  # unused after in-file writers added

# Cache the global CIFAR-10 test loader to avoid per-round re-instantiation
_GLOBAL_TEST_LOADER = None

def get_cached_global_test_loader(batch_size: int = 64):
    global _GLOBAL_TEST_LOADER
    if _GLOBAL_TEST_LOADER is None:
        _GLOBAL_TEST_LOADER = get_cifar10_test_loader(batch_size=batch_size)
    return _GLOBAL_TEST_LOADER

# Build a cluster-specific CIFAR-10 test loader by uniting train indices of member servers
def get_cluster_test_loader(cluster_servers: List[int], batch_size: int = 64):
    """Return a DataLoader over the test subset corresponding to the union of
    training shards for all servers in this cluster.

    Implementation mirrors the train->test index mapping used in fedge.task.load_data.
    Returns None if partitions or data are unavailable (no centralized fallback).
    """
    try:
        # Locate partitions mapping
        parts_json = os.getenv("PARTITIONS_JSON")
        if not parts_json:
            parts_json = str(Path().resolve() / "rounds" / "partitions.json")

        with open(parts_json, "r", encoding="utf-8") as fp:
            mapping = json.load(fp)

        # Union all training indices across all clients for the servers in this cluster
        idx_train: List[int] = []
        for sid in cluster_servers:
            server_map = mapping.get(str(sid), {})
            for _cid, indices in server_map.items():
                idx_train.extend(indices)

        if not idx_train:
            return None

        # Load CIFAR-10 datasets via torchvision-backed loader
        data = load_cifar10_hf(seed=42)
        train_size = len(data.train)
        test_size = len(data.test)

        if train_size == 0 or test_size == 0:
            return None

        # Map training indices to test indices proportionally (same rule as task.load_data)
        idx_test = [min(int(idx * test_size / train_size), test_size - 1) for idx in idx_train]

        # Deduplicate while preserving order
        seen = set()
        idx_test = [x for x in idx_test if not (x in seen or seen.add(x))]

        if not idx_test:
            return None

        test_subset = subset(data.test, idx_test)
        # Log shard sizes for diagnostics (train union size vs. resulting test shard)
        try:
            logger.info(f"[Cloud] Cluster test shard: |train_union|={len(idx_train)} -> |test|={len(idx_test)}")
        except Exception:
            pass
        return make_loader(test_subset, batch_size=batch_size, shuffle=False)

    except Exception as e:
        logger.warning(f"[Cloud] Failed to build cluster test loader; skipping eval: {e}")
        return None

def _num_bytes(params: Parameters | NDArrays) -> int:
    try:
        arrs = params if isinstance(params, list) else parameters_to_ndarrays(params)
        return int(sum(int(a.size) * int(a.itemsize) for a in arrs))
    except Exception:
        return 0

def create_signal_file(signal_path: Path, message: str) -> None:
    """Create a signal file with the given message."""
    try:
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(signal_path, "w") as f:
            f.write(f"{message}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"[Cloud Server] Failed to create signal file {signal_path}: {e}")

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# NOTE: Global evaluation at the cloud tier is disabled by design. This stub remains
# for backward compatibility but is not used anywhere.
def evaluate_global_model(server_round: int, parameters: Parameters, config: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
    logger.info("[Cloud Eval] Global evaluation disabled; returning None")
    return None

class CloudFedAvg(FedAvg):
    """CFL cloud strategy with clustering-aware aggregation and logging."""
    
    def __init__(self, **kwargs):
        # Extract cloud clustering configuration
        cloud_cluster_config = kwargs.pop('cloud_cluster_config', None)
        # Extract optional model reference for advanced algorithms
        model_ref = kwargs.pop('model', None)
        # Extract clustering configuration dict if provided
        cluster_cfg = kwargs.pop('clustering_config', None)
        
        # Do not register a cloud-level evaluate_fn; centralized eval is disabled
        super().__init__(**kwargs)
        
        # Cloud clustering configuration - enable clustering from round 1
        ccfg = cloud_cluster_config or {}
        self.cloud_cluster_config = ccfg
        self._cluster_tau = float(ccfg.get("tau", 0.7))
        self._cloud_cluster_enable = bool(ccfg.get("enable", True))  # Default enabled
        self._start_round = int(ccfg.get("start_round", 1))  # Start from round 1
        self._frequency = max(1, int(ccfg.get("frequency", 1)))  # Every round
        self._model = model_ref  # torch.nn.Module instance
        
        # Track metrics across rounds for convergence analysis
        self._previous_round_accuracy = None
        
        # Cloud clustering state
        self._current_round = 0
        self._cluster_log_data = []
        self._cluster_assignments = {}
        self._reference_set = None
        
        if self._cloud_cluster_enable:
            logger.info(f"[Cloud] Clustering enabled: start_round={self._start_round}, frequency={self._frequency}")
        else:
            logger.info(f"[Cloud] Clustering DISABLED in config: {ccfg}")
    
    def _get_server_id_from_fit_result(self, fit_result) -> Optional[int]:
        """Extract server ID from fit result metrics."""
        try:
            metrics = fit_result.metrics
            # Try multiple property names that the proxy client sends
            for prop_name in ["server_id", "sid", "node_id", "client_id", "partition_id"]:
                if prop_name in metrics:
                    return int(metrics[prop_name])
            return None
        except Exception as exc:
            logger.error(f"[Cloud Server] Failed to get server_id from fit result: {exc}")
            return None
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results; optionally perform cloud-tier clustering; save models."""
        # Use server_round directly (ignore GLOBAL_ROUND env var)
        global_round = server_round
        
        logger.info(f"[Cloud] aggregate_fit called: server_round={server_round}, global_round={global_round}, results={len(results)}, failures={len(failures)}")
        
        if failures:
            logger.warning(f"[Cloud Server] {len(failures)} servers failed during training")
        
        if not results:
            logger.warning("[Cloud Server] No results to aggregate")
            return super().aggregate_fit(server_round, results, failures)
        
        # 1) Standard FedAvg aggregation across all servers
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Convert Flower Parameters to list[ndarray] for saving/processing
        agg_list = parameters_to_ndarrays(aggregated_params) if aggregated_params is not None else None
        
        # CRITICAL: Validate aggregated parameters for NaN/inf corruption
        if agg_list is not None:
            has_nan = any(np.isnan(arr).any() for arr in agg_list)
            has_inf = any(np.isinf(arr).any() for arr in agg_list)
            
            if has_nan or has_inf:
                logger.error(f"[Cloud] CRITICAL: Aggregated model contains {'NaN' if has_nan else ''}{'/' if has_nan and has_inf else ''}{'inf' if has_inf else ''} parameters!")
                logger.error(f"[Cloud] FedAvg aggregation corrupted - using fallback weighted average")
                
                # Fallback: Simple weighted average of server models
                total_samples = sum(fit_result.num_examples for _, fit_result in results)
                if total_samples > 0:
                    weighted_sum = None
                    for _, fit_result in results:
                        weight = fit_result.num_examples / total_samples
                        server_params = parameters_to_ndarrays(fit_result.parameters)
                        
                        if weighted_sum is None:
                            weighted_sum = [weight * arr for arr in server_params]
                        else:
                            for i, arr in enumerate(server_params):
                                weighted_sum[i] += weight * arr
                    
                    # Validate fallback parameters
                    if weighted_sum is not None:
                        fallback_has_nan = any(np.isnan(arr).any() for arr in weighted_sum)
                        fallback_has_inf = any(np.isinf(arr).any() for arr in weighted_sum)
                        
                        if not fallback_has_nan and not fallback_has_inf:
                            logger.info(f"[Cloud] Fallback weighted average is clean - using as replacement")
                            agg_list = weighted_sum
                            aggregated_params = ndarrays_to_parameters(weighted_sum)
                        else:
                            logger.error(f"[Cloud] FATAL: Even fallback parameters are corrupted!")
                            return None, {}
                else:
                    logger.error(f"[Cloud] FATAL: No samples to compute fallback!")
                    return None, {}
            else:
                logger.debug(f"[Cloud] Aggregated parameters validated - no NaN/inf detected")

        # Skipping save of aggregated global model (disabled by spec)
        if agg_list is not None:
            logger.info("[Cloud] Skipping save of aggregated global model (disabled)")

        # 2) Cloud-tier clustering (weight-based) if enabled for this round
        should_cluster = (
            self._cloud_cluster_enable
            and global_round >= self._start_round
            and (global_round - self._start_round) % self._frequency == 0
        )
        
        logger.info(f"[Cloud] Clustering check: enable={self._cloud_cluster_enable}, "
                   f"round={global_round}, start={self._start_round}, freq={self._frequency}, "
                   f"should_cluster={should_cluster}")
        
        # Reset per-server cluster parameter map each round
        self._server_to_cluster_params_map: Dict[int, Parameters] = {}

        if should_cluster:
            logger.info(f"[Cloud] *** CLUSTERING TRIGGERED *** at global round {global_round} "
                        f"(start={self._start_round}, every={self._frequency})")
            
            # Extract server models and weights
            server_models = {}
            server_weights = {}
            
            for client_proxy, fit_result in results:
                server_id = self._get_server_id_from_fit_result(fit_result)
                logger.info(f"[Cloud] Processing result from server_id={server_id}, samples={fit_result.num_examples}")
                if server_id is not None:
                    server_models[server_id] = parameters_to_ndarrays(fit_result.parameters)
                    server_weights[server_id] = fit_result.num_examples
                    logger.info(f"[Cloud] Server {server_id}: {fit_result.num_examples} samples, model params={len(server_models[server_id])}")
                else:
                    logger.warning(f"[Cloud] Failed to extract server_id from fit result")
            
            logger.info(f"[Cloud] Total server models collected: {len(server_models)} (need >=2 for clustering)")
            logger.info(f"[Cloud] Server IDs collected: {list(server_models.keys())}")
            
            if len(server_models) >= 2:
                # Perform clustering using cluster_utils
                from fedge.cluster_utils import cifar10_weight_clustering
                
                try:
                    # Convert server_models dict to list for clustering function
                    server_ids = sorted(server_models.keys())
                    server_weights_list = [server_models[sid] for sid in server_ids]
                    
                    # Read tau from cloud cluster config
                    tau = getattr(self, "_cluster_tau", 0.7)
                    logger.info(f"[Cloud] Using tau = {tau} from config for clustering")
                    
                    # Convert aggregated_params to list if it's Parameters type
                    if aggregated_params is not None:
                        global_weights = parameters_to_ndarrays(aggregated_params)
                    else:
                        global_weights = server_weights_list[0]  # Use first as fallback
                    
                    labels, S, tau_sim = cifar10_weight_clustering(
                        server_weights_list,
                        global_weights,
                        None,  # reference_imgs not needed for weight-based clustering
                        global_round,  # Use global_round consistently
                        tau
                    )
                    cluster_labels_array = labels
                    
                    # Surface pairwise similarities in cloud logs for debugging
                    for i in range(S.shape[0]):
                        logger.info(f"[Cloud] S[{i}]: " + " ".join(f"{v:.3f}" for v in S[i]))
                    logger.info(f"[Cloud] similarity_threshold used: {tau_sim:.3f}")
                    
                    # Convert cluster labels array to dict mapping server_id -> cluster_label
                    cluster_map = {server_ids[i]: int(cluster_labels_array[i]) for i in range(len(server_ids))}
                    unique_labels = set(cluster_map.values())
                    
                    logger.info(f"[Cloud] Clustering results: {len(unique_labels)} clusters")
                    logger.info(f"[Cloud] Cluster assignments: {cluster_map}")
                    
                    # Note: legacy metrics/cloud path removed; cluster mapping is saved
                    # under rounds/... and metrics written to metrics/seed_{seed}/cloud/ later.
                    
                    # Save cluster artifacts using rounds directory structure
                    cl_dir = Path().resolve() / "rounds" / f"round_{global_round}" / "cloud"
                    cl_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save cluster mapping in canonical location and mirror to models/
                    cluster_json_path = cl_dir / f"clusters_g{global_round}.json"
                    with open(cluster_json_path, "w", encoding="utf-8") as fp:
                        json.dump({"round": global_round, "assignments": cluster_map}, fp, indent=2)
                    logger.info(f"[Cloud] Saved cluster mapping to {cluster_json_path}")
                    
                    # Save similarity matrix to file for analysis
                    similarity_matrix_path = cl_dir / f"similarity_matrix_g{global_round}.csv"
                    np.savetxt(similarity_matrix_path, S, delimiter=',', fmt='%.6f')
                    logger.info(f"[Cloud] Saved similarity matrix to {similarity_matrix_path}")
                    
                    # Save pairwise similarities as detailed CSV
                    pairwise_csv_path = cl_dir / f"pairwise_similarities_g{global_round}.csv"
                    with open(pairwise_csv_path, 'w', newline='') as f:
                        import csv
                        writer = csv.writer(f)
                        writer.writerow(['server_i', 'server_j', 'similarity', 'above_threshold'])
                        for i in range(len(server_ids)):
                            for j in range(i+1, len(server_ids)):
                                sim_val = S[i, j]
                                above_thresh = sim_val >= tau
                                writer.writerow([server_ids[i], server_ids[j], f'{sim_val:.6f}', above_thresh])
                    logger.info(f"[Cloud] Saved pairwise similarities to {pairwise_csv_path}")
                    
                    # Mirror to models/ directory for compatibility
                    models_dir = Path().resolve() / "models"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cluster_json_path, models_dir / f"clusters_g{global_round}.json")
                    
                    # Save cluster-specific models
                    for lab in unique_labels:
                        # Get servers in this cluster
                        cluster_servers = [sid for sid, label in cluster_map.items() if label == lab]
                        
                        # Weighted average within cluster
                        total_weight = sum(server_weights.get(sid, 1) for sid in cluster_servers)
                        sums = None
                        
                        for sid in cluster_servers:
                            if sid in server_models:
                                weight = server_weights.get(sid, 1) / total_weight
                                model_arrays = server_models[sid]
                                
                                if sums is None:
                                    sums = [weight * arr for arr in model_arrays]
                                else:
                                    for i, arr in enumerate(model_arrays):
                                        sums[i] += weight * arr
                        
                        if sums is not None:
                            # Save as plain list for evaluator compatibility
                            cluster_path = cl_dir / f"model_cluster{lab}_g{global_round}.pkl"
                            with open(cluster_path, "wb") as f:
                                pickle.dump(sums, f, protocol=pickle.HIGHEST_PROTOCOL)
                            logger.info(f"[Cloud] Saved cluster {lab} model to {cluster_path}")
                            
                            # Mirror to models/ directory
                            shutil.copy2(cluster_path, models_dir / cluster_path.name)

                            # Build and store cluster Parameters for downlink
                            try:
                                cluster_params = ndarrays_to_parameters([arr for arr in sums])
                                for sid in cluster_servers:
                                    self._server_to_cluster_params_map[int(sid)] = cluster_params
                                logger.info(f"[Cloud] Prepared cluster parameters for servers {cluster_servers}")
                            except Exception as e:
                                logger.error(f"[Cloud] Failed to convert cluster head to Parameters: {e}")

                    logger.info(f"[Cloud] *** CLUSTERING COMPLETED *** @ global round {global_round}: K={len(unique_labels)} labels={cluster_map}")
                    
                    # Log clustering results for verification
                    for lab in unique_labels:
                        cluster_servers = [sid for sid, label in cluster_map.items() if label == lab]
                        logger.info(f"[Cloud] Cluster {lab}: servers {cluster_servers}")
                    
                except Exception as e:
                    logger.error(f"[Cloud] Clustering failed: {e}, using standard aggregation")
                    import traceback
                    logger.error(f"[Cloud] Clustering error traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[Cloud] Not enough servers for clustering: {len(server_models)} < 2")
        else:
            logger.info(f"[Cloud] Clustering skipped for round {global_round}")

        logger.info(f"[Cloud] Aggregation completed for global round {global_round}")
        
        # Compute and write cluster metrics (consolidated CSV for all rounds)
        try:
            # Load config to determine seed for metrics path
            cfg = toml.load("pyproject.toml")
            seed = int(cfg.get("tool", {}).get("flwr", {}).get("hierarchy", {}).get("dirichlet", {}).get("seed", 0))
            base_metrics_dir = Path().resolve() / "metrics" / f"seed_{seed}" / "cloud"
            base_metrics_dir.mkdir(parents=True, exist_ok=True)

            # Build server_id -> num_examples map from results for cluster weighting
            sid_to_examples: Dict[int, int] = {}
            for _, fit_result in results:
                sid = self._get_server_id_from_fit_result(fit_result)
                if sid is not None:
                    sid_to_examples[sid] = int(getattr(fit_result, "num_examples", 0))

            # Attempt to load cluster assignments produced earlier this round
            cl_dir = Path().resolve() / "rounds" / f"round_{global_round}" / "cloud"
            cluster_json = cl_dir / f"clusters_g{global_round}.json"

            cluster_rows: List[Dict[str, Any]] = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Check if clustering happened this round
            if cluster_json.exists():
                # Multi-cluster case: evaluate each cluster head
                with open(cluster_json, "r", encoding="utf-8") as fp:
                    cluster_data = json.load(fp)
                cluster_map = {int(k): int(v) for k, v in cluster_data.get("assignments", {}).items()}
                unique_labels = sorted(set(cluster_map.values()))

                for lab in unique_labels:
                    cluster_servers = [sid for sid, label in cluster_map.items() if label == lab]
                    num_servers = len(cluster_servers)
                    num_examples = int(sum(sid_to_examples.get(sid, 0) for sid in cluster_servers))

                    cluster_path = cl_dir / f"model_cluster{lab}_g{global_round}.pkl"
                    loss_c = float("nan")
                    acc_c = float("nan")
                    test_loader = get_cluster_test_loader(cluster_servers, batch_size=64)
                    if cluster_path.exists() and test_loader is not None:
                        with open(cluster_path, "rb") as f:
                            cluster_nd = pickle.load(f)
                        model = Net()
                        model.to(device)
                        set_weights(model, cluster_nd)
                        loss_c, acc_c = test(model, test_loader, device)

                    cluster_rows.append({
                        "round": global_round,
                        "cluster_id": lab,
                        "num_servers": num_servers,
                        "num_examples": num_examples,
                        "cluster_accuracy": float(acc_c),
                        "cluster_loss": float(loss_c),
                    })
            else:
                # No clustering this round: treat all servers as single cluster (cluster_id=0)
                all_server_ids = list(sid_to_examples.keys())
                total_examples = sum(sid_to_examples.values())
                
                # Single cluster evaluation using aggregated parameters
                loss_c = float("nan")
                acc_c = float("nan")
                if aggregated_params is not None:
                    # Use union of all servers' data for evaluation
                    test_loader = get_cluster_test_loader(all_server_ids, batch_size=64)
                    if test_loader is not None:
                        model = Net()
                        model.to(device)
                        set_weights(model, parameters_to_ndarrays(aggregated_params))
                        loss_c, acc_c = test(model, test_loader, device)
                
                cluster_rows.append({
                    "round": global_round,
                    "cluster_id": 0,  # single cluster
                    "num_servers": len(all_server_ids),
                    "num_examples": total_examples,
                    "cluster_accuracy": float(acc_c),
                    "cluster_loss": float(loss_c),
                })

            # Calculate comprehensive aggregated metrics (one row per round)
            if cluster_rows:
                total_examples_all = sum(row["num_examples"] for row in cluster_rows)
                total_servers_all = sum(row["num_servers"] for row in cluster_rows)
                num_clusters = len(cluster_rows)
                
                # Extract valid accuracy and loss values
                valid_accuracies = [row["cluster_accuracy"] for row in cluster_rows if not np.isnan(row["cluster_accuracy"])]
                valid_losses = [row["cluster_loss"] for row in cluster_rows if not np.isnan(row["cluster_loss"])]
                valid_examples = [row["num_examples"] for row in cluster_rows if not np.isnan(row["cluster_accuracy"])]
                
                if valid_accuracies and total_examples_all > 0:
                    # Weighted average accuracy and loss
                    weighted_acc = sum(acc * examples for acc, examples in zip(valid_accuracies, valid_examples)) / sum(valid_examples)
                    weighted_loss = sum(loss * examples for loss, examples in zip(valid_losses, valid_examples)) / sum(valid_examples) if valid_losses else float("nan")
                    
                    # Standard deviation across clusters (unweighted for cluster variance)
                    if len(valid_accuracies) > 1:
                        acc_std = float(np.std(valid_accuracies, ddof=1))
                        loss_std = float(np.std(valid_losses, ddof=1)) if len(valid_losses) > 1 else float("nan")
                    else:
                        acc_std = 0.0
                        loss_std = 0.0
                    
                    # Min/Max for range analysis
                    acc_min, acc_max = float(min(valid_accuracies)), float(max(valid_accuracies))
                    loss_min, loss_max = (float(min(valid_losses)), float(max(valid_losses))) if valid_losses else (float("nan"), float("nan"))
                    
                    # Convergence indicator (accuracy improvement from previous round)
                    if self._previous_round_accuracy is not None:
                        convergence_rate = float(weighted_acc - self._previous_round_accuracy)
                    else:
                        convergence_rate = 0.0  # First round
                    
                    # Update previous round accuracy for next iteration
                    self._previous_round_accuracy = weighted_acc
                    
                    # Generalization gap placeholder (would need train accuracy to compute)
                    generalization_gap = 0.0  # Placeholder
                    
                else:
                    weighted_acc = weighted_loss = float("nan")
                    acc_std = loss_std = float("nan")
                    acc_min = acc_max = loss_min = loss_max = float("nan")
                    convergence_rate = generalization_gap = 0.0
                
                # Create single consolidated row per round
                consolidated_row = {
                    "round": global_round,
                    "num_clusters": num_clusters,
                    "num_servers": total_servers_all,
                    "num_examples": total_examples_all,
                    "agg_accuracy": float(weighted_acc),
                    "agg_loss": float(weighted_loss),
                    "acc_std": float(acc_std),
                    "loss_std": float(loss_std),
                    "acc_min": float(acc_min),
                    "acc_max": float(acc_max),
                    "loss_min": float(loss_min),
                    "loss_max": float(loss_max),
                    "convergence_rate": float(convergence_rate),
                    "generalization_gap": float(generalization_gap),
                    # Store individual cluster values as comma-separated strings for reference
                    "cluster_accuracies": ",".join(f"{acc:.6f}" for acc in valid_accuracies) if valid_accuracies else "",
                    "cluster_losses": ",".join(f"{loss:.6f}" for loss in valid_losses) if valid_losses else "",
                }

            # Write to consolidated CSV (append mode) - one row per round
            clusters_csv = base_metrics_dir / "cloud_metrics.csv"
            import csv as _csv
            write_header = not clusters_csv.exists()
            with open(clusters_csv, "a", newline="") as fcsv:
                writer = _csv.DictWriter(fcsv, fieldnames=[
                    "round", "num_clusters", "num_servers", "num_examples",
                    "agg_accuracy", "agg_loss", "acc_std", "loss_std",
                    "acc_min", "acc_max", "loss_min", "loss_max",
                    "convergence_rate", "generalization_gap",
                    "cluster_accuracies", "cluster_losses"
                ])
                if write_header:
                    writer.writeheader()
                if cluster_rows:  # Only write if we have data
                    writer.writerow(consolidated_row)
            logger.info(f"[Cloud] Saved consolidated cloud metrics for round {global_round} → {clusters_csv}")
        except Exception as e:
            logger.error(f"[Cloud] Failed to compute/write cluster metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # Create per-round completion signal for orchestrator synchronization
        signal_dir = Path().resolve() / "signals"
        signal_dir.mkdir(exist_ok=True)
        completion_signal = signal_dir / f"cloud_round_{server_round}_completed.signal"
        create_signal_file(completion_signal, f"Cloud round {server_round} completed")
        logger.info(f"[Cloud] Created completion signal: {completion_signal}")
        
        return aggregated_params, metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Distribute cluster-specific parameters to each server (proxy)."""
        # Get default FitIns from parent (client selection, config, etc.)
        fit_cfgs = super().configure_fit(server_round, parameters, client_manager)

        mapping: Dict[int, Parameters] = getattr(self, "_server_to_cluster_params_map", {}) or {}
        if not mapping:
            # No cluster-specific params this round; fall back to default behavior
            logger.info(f"[Cloud] No cluster-specific parameters for round {server_round}; using default aggregated params")
            return fit_cfgs

        # Replace parameters per client based on server_id from client proxy id
        replaced = 0
        for idx, (client, fit_ins) in enumerate(fit_cfgs):
            sid_int = None
            try:
                cid_str = getattr(client, "cid", "")
                if cid_str and cid_str.isdigit():
                    sid_int = int(cid_str)
                else:
                    # Extract first integer sequence from cid if present
                    import re
                    m = re.search(r"(\d+)", str(cid_str))
                    if m:
                        sid_int = int(m.group(1))
            except Exception:
                sid_int = None

            if sid_int is not None and sid_int in mapping:
                fit_ins.parameters = mapping[sid_int]
                fit_ins.config = dict(fit_ins.config)
                fit_ins.config["server_id"] = sid_int
                replaced += 1

        logger.info(f"[Cloud] Distributed cluster-specific parameters to {replaced} servers in round {server_round}")
        return fit_cfgs
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results and compute agg_acc correctly."""
        try:
            if failures:
                logger.warning(f"[Cloud] Evaluation failures: {len(failures)}")
                return None, {"agg_acc": 0.0}
            
            if not results:
                logger.warning(f"[Cloud] No evaluation results to aggregate")
                return None, {"agg_acc": 0.0}
            
            # Compute weighted average accuracy
            total_examples = 0
            weighted_accuracy_sum = 0.0
            
            for _, eval_res in results:
                num_examples = getattr(eval_res, "num_examples", 0)
                accuracy = (eval_res.metrics or {}).get("accuracy", 0.0)
                
                weighted_accuracy_sum += accuracy * num_examples
                total_examples += num_examples
            
            agg_acc = (weighted_accuracy_sum / total_examples) if total_examples > 0 else 0.0
            
            # Call parent aggregate_evaluate for loss aggregation
            parent_result = super().aggregate_evaluate(server_round, results, failures)
            if parent_result is not None:
                loss, parent_metrics = parent_result
                parent_metrics = parent_metrics or {}
                parent_metrics["agg_acc"] = agg_acc
                return loss, parent_metrics
            else:
                return None, {"agg_acc": agg_acc}
                
        except Exception as e:
            logger.error(f"[Cloud Eval] Evaluation failed: {e}")
            return None, {"agg_acc": 0.0}

def run_server():
    """Run long-running cloud server for all global rounds with dynamic clustering."""
    logger.info("[Cloud Server] Starting long-running cloud aggregation server")
    
    # Create signal directory
    signal_dir = Path().resolve() / "signals"
    signal_dir.mkdir(exist_ok=True)
    
    # Get total number of global rounds from environment
    total_rounds = int(os.getenv('TOTAL_GLOBAL_ROUNDS', '3'))
    logger.info(f"[Cloud Server] Will handle {total_rounds} global rounds")
    
    try:
        # Log environment variables set by orchestrator
        num_servers = os.getenv('NUM_SERVERS', '3')
        logger.info(f"[Cloud Server] Starting for {total_rounds} rounds with {num_servers} servers")
        
        # Load configuration
        config_path = Path("pyproject.toml")
        if not config_path.exists():
            raise FileNotFoundError("pyproject.toml not found")
        
        # Log working directory and models path for path verification
        cwd = Path().resolve()
        logger.info(f"[Cloud Server] Working directory: {cwd}")
        logger.info(f"[Cloud Server] Models directory: {cwd / 'models'}")

        config = toml.load(config_path)
        
        # Extract configuration
        hierarchy_config = config.get("tool", {}).get("flwr", {}).get("hierarchy", {})
        optimizer_config = config.get("tool", {}).get("flwr", {}).get("optimizer", {})
        cluster_config = config.get("tool", {}).get("flwr", {}).get("cluster", {})
        cloud_cluster_cfg = config.get("tool", {}).get("flwr", {}).get("cloud_cluster", {})
        
        # CRITICAL FIX: Load previous round's model or create fresh parameters
        global_round = int(os.getenv('GLOBAL_ROUND', '1'))
        initial_parameters = None
        # Initialize model upfront to avoid undefined variable
        init_model = Net()
        
        # Try to load previous round's global model
        if global_round > 1:
            prev_round = global_round - 1
            candidate_paths = [
                cwd / "rounds" / f"round_{prev_round}" / "global" / "model.pkl",
                cwd / "models" / f"model_global_g{prev_round}.pkl",
                cwd / "models" / f"global_model_round_{prev_round}.pkl"
            ]
            
            for model_path in candidate_paths:
                if model_path.exists():
                    try:
                        with open(model_path, "rb") as f:
                            loaded_data = pickle.load(f)
                            if isinstance(loaded_data, tuple):
                                ndarrays = loaded_data[0]
                            else:
                                ndarrays = loaded_data
                        initial_parameters = ndarrays_to_parameters(ndarrays)
                        # Also set weights on init_model for consistency
                        set_weights(init_model, ndarrays)
                        logger.info(f"[Cloud Server] Loaded trained model from round {prev_round}")
                        break
                    except Exception as e:
                        logger.warning(f"[Cloud Server] Failed to load {model_path}: {e}")
        
        # If no previous model found, create fresh parameters
        if initial_parameters is None:
            initial_parameters = ndarrays_to_parameters(get_weights(init_model))
            logger.info(f"[Cloud Server] Created fresh model parameters for round {global_round}")

        # Get number of servers from environment (set by orchestrator)
        num_servers = int(os.getenv('NUM_SERVERS', '3'))
        
        # Use CloudFedAvg for CIFAR-10 with optional cloud clustering
        strategy = CloudFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,  # disable cloud-tier evaluation requests
            min_fit_clients=num_servers,
            min_evaluate_clients=0,  # no eval clients required
            min_available_clients=num_servers,
            initial_parameters=initial_parameters,
            clustering_config=cluster_config,
            cloud_cluster_config=cloud_cluster_cfg,
            model=init_model,
        )
        # Enforce "all servers must participate"
        strategy.accept_failures = False
        
        # Start server to handle all global rounds
        logger.info(f"[Cloud Server] Starting on port {hierarchy_config.get('cloud_port', 6000)}")
        logger.info(f"[Cloud Server] Expecting {num_servers} proxy clients per round")
        
        # Create start signal file BEFORE starting server
        create_signal_file(signal_dir / "cloud_started.signal", "Cloud server started")
        logger.info("[Cloud Server] Start signal created")
        
        # Start server with proper configuration - handle ALL rounds in one go
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            logger.info(f"[Cloud Server] Starting Flower server for {total_rounds} rounds")
            history = start_server(
                server_address=f"0.0.0.0:{hierarchy_config.get('cloud_port', 6000)}",
                config=ServerConfig(
                    num_rounds=total_rounds,  # Handle ALL rounds in one server instance
                    round_timeout=300.0,  # 5 minutes timeout per round
                ),
                strategy=strategy
            )
        
        logger.info(f"[Cloud Server] All {total_rounds} rounds completed")
        return history
        
    except Exception as e:
        logger.exception(f"[Cloud Server] Unhandled error: {e}")
        create_signal_file(signal_dir / "cloud_error.signal", f"{e!r}")
        raise
    finally:
        # Create final completion signal after all rounds
        create_signal_file(signal_dir / "cloud_all_rounds_completed.signal", "Cloud server completed all rounds")

if __name__ == "__main__":
    run_server()
