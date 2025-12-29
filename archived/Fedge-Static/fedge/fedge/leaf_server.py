# fedge/leaf_server.py

import argparse
import pickle
import os
import sys
import time
import signal
import warnings
import logging
import csv
import gc
from typing import List, Tuple, Optional, Any
from pathlib import Path
import toml
from fedge.utils.agg import sum_metrics

from fedge.utils import fs

from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg

from fedge.task import Net, get_weights, set_weights, load_data, test, get_cifar10_test_loader
import torch
import json
import psutil
import numpy as np
from collections import defaultdict, deque, Counter
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

# Fix Windows encoding issues
if sys.platform == "win32":
    import codecs
    # Set console encoding to UTF-8 to handle Unicode characters
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Suppress Python deprecation warnings in flwr module
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
# Elevate Flower, ECE, gRPC logger levels to ERROR to hide warning logs
for name in ("flwr", "ece", "grpc"): logging.getLogger(name).setLevel(logging.ERROR)
# Drop printed 'DEPRECATED FEATURE' messages from stdout/stderr
class _DropDeprecated:
    def __init__(self, out): self._out = out
    def write(self, txt):
        if "DEPRECATED FEATURE" not in txt: self._out.write(txt)
    def flush(self): self._out.flush()

sys.stdout = _DropDeprecated(sys.stdout)
sys.stderr = _DropDeprecated(sys.stderr)


class MultiLevelMetrics:
    """Tracks comprehensive metrics for multi-level hierarchical clustering"""
    
    def __init__(self, server_id: int):
        self.server_id = server_id
        self.communication_metrics = []
        self.computation_metrics = []
        self.performance_metrics = []
        self.cluster_metrics = []
        self.memory_metrics = []
        
    def log_communication(self, round_time: float, num_clients: int, bytes_transferred: int):
        """Log communication overhead metrics"""
        self.communication_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'round_time': round_time,
            'num_clients': num_clients,
            'bytes_transferred': bytes_transferred,
            'avg_time_per_client': round_time / max(num_clients, 1)
        })
        
    def log_computation(self, clustering_time: float, aggregation_time: float):
        """Log computation cost metrics"""
        self.computation_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'clustering_time': clustering_time,
            'aggregation_time': aggregation_time,
            'total_computation_time': clustering_time + aggregation_time
        })
        
    def log_performance(self, cluster_accuracies: dict, cluster_sizes: dict, server_accuracy: float):
        """Log performance metrics for clusters and server"""
        self.performance_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'cluster_accuracies': cluster_accuracies.copy(),
            'cluster_sizes': cluster_sizes.copy(),
            'server_accuracy': server_accuracy,
            'weighted_cluster_accuracy': sum(acc * cluster_sizes.get(cluster, 0) 
                                           for cluster, acc in cluster_accuracies.items()) / 
                                        max(sum(cluster_sizes.values()), 1)
        })
        
    def log_memory_usage(self):
        """Log current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'memory_rss_mb': memory_info.rss / 1024 / 1024,
            'memory_vms_mb': memory_info.vms / 1024 / 1024,
            'cpu_percent': process.cpu_percent()
        })
        
    def save_metrics_optimized(self, round_num: int):
        """Save all metrics using optimized consolidated storage"""
        try:
            # Prepare server metrics
            latest_perf = self.performance_metrics[-1] if self.performance_metrics else {}
            server_metrics = {
                'server_accuracy': latest_perf.get('server_accuracy', 0.0),
                'server_loss': latest_perf.get('server_loss', 0.0),
                'local_samples': latest_perf.get('local_samples', 0),
                'aggregation_weight': latest_perf.get('aggregation_weight', 0.0),
                'cluster_id': getattr(self, 'cluster_id', -1),
                'drift_correction': latest_perf.get('drift_correction', 0.0)
            }
            
            # Use consolidated metrics storage
            from fedge.utils.fs_optimized import save_server_metrics_optimized
            save_server_metrics_optimized(
                self.project_root, 
                round_num, 
                self.server_id,
                server_metrics,
                self.communication_metrics or [],
                self.communication_metrics or []  # Using comm metrics for both client and comm data
            )
            
            logger.info(f"[Server {self.server_id}] Successfully saved metrics using consolidated storage")
            
        except Exception as e:
            logger.error(f"[Server {self.server_id}] Failed to save metrics: {e}")
            # Don't re-raise to prevent crashing the server
    
    def _save_communication_csv(self, output_dir: Path, round_num: int):
        """Save communication metrics in CSV format for plotting"""
        comm_csv_path = output_dir / "communication_metrics.csv"
        
        # Check if file exists to determine if we need headers
        write_header = not comm_csv_path.exists()
        
        with open(comm_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow([
                    'global_round', 'server_id', 'round_time', 'num_clients', 
                    'bytes_transferred', 'avg_time_per_client', 'bytes_per_client',
                    'cumulative_bytes', 'cumulative_time'
                ])
            
            # Calculate cumulative metrics
            cumulative_bytes = sum(m.get('bytes_transferred', 0) for m in self.communication_metrics)
            cumulative_time = sum(m.get('round_time', 0) for m in self.communication_metrics)
            
            if self.communication_metrics:
                latest_comm = self.communication_metrics[-1]
                bytes_transferred = latest_comm.get('bytes_transferred', 0)
                num_clients = latest_comm.get('num_clients', 1)
                
                writer.writerow([
                    round_num, self.server_id, 
                    latest_comm.get('round_time', 0),
                    num_clients,
                    bytes_transferred,
                    latest_comm.get('avg_time_per_client', 0),
                    bytes_transferred / max(num_clients, 1),
                    cumulative_bytes,
                    cumulative_time
                ])
    
    def _save_performance_csv(self, output_dir: Path, round_num: int):
        """Save performance metrics in CSV format for plotting"""
        perf_csv_path = output_dir / "performance_metrics.csv"
        
        write_header = not perf_csv_path.exists()
        
        with open(perf_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow([
                    'global_round', 'server_id', 'server_accuracy', 
                    'weighted_cluster_accuracy', 'num_clusters'
                ])
            
            if self.performance_metrics:
                latest_perf = self.performance_metrics[-1]
                cluster_accuracies = latest_perf.get('cluster_accuracies', {})
                
                writer.writerow([
                    round_num, self.server_id,
                    latest_perf.get('server_accuracy', 0),
                    latest_perf.get('weighted_cluster_accuracy', 0),
                    len(cluster_accuracies)
                ])
    
    def _save_timing_csv(self, output_dir: Path, round_num: int):
        """Save timing metrics in CSV format for plotting"""
        timing_csv_path = output_dir / "timing_metrics.csv"
        
        write_header = not timing_csv_path.exists()
        
        with open(timing_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow([
                    'global_round', 'server_id', 'clustering_time', 
                    'aggregation_time', 'total_computation_time',
                    'cumulative_computation_time'
                ])
            
            # Calculate cumulative computation time
            cumulative_computation = sum(m.get('total_computation_time', 0) for m in self.computation_metrics)
            
            if self.computation_metrics:
                latest_comp = self.computation_metrics[-1]
                
                writer.writerow([
                    round_num, self.server_id,
                    latest_comp.get('clustering_time', 0),
                    latest_comp.get('aggregation_time', 0),
                    latest_comp.get('total_computation_time', 0),
                    cumulative_computation
                ])


class ClientClusterManager:
    """Manages client clustering based on label distribution skew for multi-level hierarchical FL"""
    
    def __init__(self, server_id: int, config: dict):
        self.server_id = server_id
        self.config = config
        self.client_label_distributions = {}  # Store actual label distributions per client
        self.label_clusters = {}  # Dynamic clusters based on label similarity
        self.cluster_assignments = {}  # Client to cluster assignments
        self.clustering_initialized = False
        
        # Configuration parameters - require explicit TOML values, no defaults
        required_params = ['similarity_threshold', 'min_cluster_size', 'max_clusters']
        for param in required_params:
            if param not in config:
                raise ValueError(f"{param} must be specified in [tool.flwr.cluster.multi_level] section of pyproject.toml")
        
        self.similarity_threshold = float(config['similarity_threshold'])
        self.min_cluster_size = int(config['min_cluster_size'])
        self.max_clusters = int(config['max_clusters'])
        self.log_cluster_details = config.get('log_cluster_details', True)  # Optional logging param
        self.log_kl_divergences = config.get('log_kl_divergences', True)   # Optional logging param
        
        logger.info(f"[Server {server_id}] ClientClusterManager initialized with label-based clustering")
        logger.info(f"[Server {server_id}] Clustering config: threshold={self.similarity_threshold}, min_size={self.min_cluster_size}, max_clusters={self.max_clusters}")
        
    def analyze_client_label_distributions(self, client_results: List[Tuple[str, Any]]):
        """Analyze actual label distributions from client training metrics - DETERMINISTIC VERSION"""
        # CRITICAL: Clustering should only happen ONCE per server instance!
        if self.clustering_initialized:
            logger.info(f"[Server {self.server_id}] ✅ DETERMINISTIC: Clustering already initialized, skipping re-clustering")
            return
        
        logger.info(f"[Server {self.server_id}] 🔄 DETERMINISTIC: Initializing client clustering (ONCE ONLY)")
        
        for client_id, fit_res in client_results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                # Extract label distribution from client metrics
                label_dist = fit_res.metrics.get('label_distribution', {})
                if not label_dist:
                    # Fallback: estimate from client data partition (now deterministic with caching)
                    label_dist = self._estimate_label_distribution(client_id)
                
                self.client_label_distributions[client_id] = label_dist
                logger.info(f"[Server {self.server_id}] Client {client_id} label distribution computed")
        
        # Perform clustering based on label similarity - ONCE ONLY
        logger.info(f"[Server {self.server_id}] 🎯 DETERMINISTIC: Performing label-based clustering (FINAL)")
        self._perform_label_based_clustering()
        
        # Mark as initialized - NO MORE CLUSTERING!
        self.clustering_initialized = True
        logger.info(f"[Server {self.server_id}] ✅ DETERMINISTIC: Client clustering completed and LOCKED")
    
    def _estimate_label_distribution(self, client_id: str):
        """Estimate label distribution for client based on partition data - DETERMINISTIC VERSION"""
        # Extract numeric client ID from consistent client_X format
        if client_id.startswith("client_"):
            client_num = int(client_id.split("_")[1])
        else:
            # Fallback for any other format
            client_num = 0
            logger.warning(f"[Server {self.server_id}] Unexpected client_id format: {client_id}, using client_num=0")
        
        # Check cache first - clustering should be deterministic!
        cache_key = f"server_{self.server_id}_client_{client_num}"
        if hasattr(self, '_label_dist_cache') and cache_key in self._label_dist_cache:
            logger.info(f"[Server {self.server_id}] Using cached label distribution for {client_id}")
            return self._label_dist_cache[cache_key]
        
        # Initialize cache if not exists
        if not hasattr(self, '_label_dist_cache'):
            self._label_dist_cache = {}
        
        try:
            from fedge.task import load_data
            import torch
            from collections import Counter
            # Use simple client_num as partition_id since our JSON structure is server_id -> client_id
            partition_id = client_num
            
            # Load training data for this client with FIXED SEED for deterministic ordering
            torch.manual_seed(42 + self.server_id * 10 + partition_id)  # Deterministic seed per client
            # Get dataset_flag from environment (passed by orchestrator)
            dataset_flag = "cifar10"
            # Use a smaller batch size for label distribution estimation
            estimation_batch_size = self.config.get('estimation_batch_size', 32)
            trainloader, _, _ = load_data(dataset_flag, partition_id, 30, batch_size=estimation_batch_size, server_id=self.server_id)
            
            # Count label frequencies - USE FULL DATASET for deterministic results
            label_counts = Counter()
            total_samples = 0
            
            logger.info(f"[Server {self.server_id}] Computing FULL label distribution for {client_id} (partition {partition_id})")
            
            for batch_idx, (_, labels) in enumerate(trainloader):
                # REMOVED: if batch_idx > 10: break  # This was causing non-deterministic behavior!
                for label in labels:
                    label_counts[int(label.item())] += 1
                    total_samples += 1
            
            # Convert to probability distribution
            if total_samples > 0:
                label_dist = {str(k): v/total_samples for k, v in label_counts.items()}
                logger.info(f"[Server {self.server_id}] {client_id} label distribution: {total_samples} samples, top classes: {sorted(label_dist.items(), key=lambda x: x[1], reverse=True)[:3]}")
            else:
                # Hard failure: no fallback uniform distribution
                raise RuntimeError(f"[Server {self.server_id}] {client_id} has no training samples - cannot proceed with clustering")
            
            # Cache the result for future use
            self._label_dist_cache[cache_key] = label_dist
            logger.info(f"[Server {self.server_id}] Cached label distribution for {client_id}")
            
            return label_dist
            
        except Exception as e:
            logger.error(f"[Server {self.server_id}] Could not estimate label distribution for {client_id}: {e}")
            # Hard failure: no fallback uniform distribution
            raise RuntimeError(f"[Server {self.server_id}] Label distribution estimation failed for {client_id} and cannot proceed: {e}") from e
    
    def _perform_label_based_clustering(self):
        """Cluster clients based on label distribution similarity using KL divergence"""
        if not self.client_label_distributions:
            return
            
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        from sklearn.cluster import AgglomerativeClustering
        
        client_ids = list(self.client_label_distributions.keys())
        n_clients = len(client_ids)
        
        if n_clients < 2:
            # Single client - create one cluster
            self.label_clusters = {'cluster_0': client_ids}
            for client_id in client_ids:
                self.cluster_assignments[client_id] = 'cluster_0'
            return
        
        # Create distance matrix based on KL divergence of label distributions
        distance_matrix = np.zeros((n_clients, n_clients))
        kl_divergences = {}  # Store for logging
        
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i != j:
                    dist_i = self.client_label_distributions[client_i]
                    dist_j = self.client_label_distributions[client_j]
                    
                    # Calculate symmetric KL divergence
                    kl_div = self._calculate_kl_divergence(dist_i, dist_j)
                    distance_matrix[i][j] = kl_div
                    
                    # Store for logging
                    if self.log_kl_divergences:
                        kl_divergences[f"{client_i}-{client_j}"] = kl_div
        
        # Log KL divergences if enabled
        if self.log_kl_divergences and kl_divergences:
            logger.info(f"[Server {self.server_id}] KL divergences between clients:")
            for pair, kl_div in sorted(kl_divergences.items()):
                logger.info(f"  - {pair}: {kl_div:.4f}")
        
        # Perform hierarchical clustering
        try:
            n_clusters = min(self.max_clusters, max(2, n_clients // self.min_cluster_size))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Organize clusters
            self.label_clusters = {}
            for i, label in enumerate(cluster_labels):
                cluster_name = f'cluster_{label}'
                if cluster_name not in self.label_clusters:
                    self.label_clusters[cluster_name] = []
                self.label_clusters[cluster_name].append(client_ids[i])
                self.cluster_assignments[client_ids[i]] = cluster_name
                
        except Exception as e:
            logger.error(f"[Server {self.server_id}] Clustering failed: {e}")
            # Hard failure: no fallback single cluster
            raise RuntimeError(f"[Server {self.server_id}] Client clustering failed and cannot proceed: {e}") from e
        
        if self.log_cluster_details:
            logger.info(f"[Server {self.server_id}] Label-based clustering completed:")
            for cluster_name, clients in self.label_clusters.items():
                logger.info(f"  - {cluster_name}: {len(clients)} clients {clients}")
                
                # Log label distributions for each cluster if detailed logging enabled
                if len(clients) > 1:
                    logger.info(f"    Cluster {cluster_name} label distributions:")
                    for client_id in clients:
                        if client_id in self.client_label_distributions:
                            dist = self.client_label_distributions[client_id]
                            # Show top 3 labels for brevity
                            top_labels = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
                            logger.info(f"      {client_id}: {top_labels}")
    
    def _calculate_kl_divergence(self, dist1: dict, dist2: dict):
        """Calculate symmetric KL divergence between two label distributions"""
        import numpy as np
        
        # Get all possible labels
        all_labels = set(dist1.keys()) | set(dist2.keys())
        
        # Convert to arrays with smoothing
        eps = 1e-8
        p = np.array([dist1.get(label, eps) for label in sorted(all_labels)])
        q = np.array([dist2.get(label, eps) for label in sorted(all_labels)])
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Symmetric KL divergence
        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))
        
        return (kl_pq + kl_qp) / 2.0
    
    def cluster_clients_by_characteristics(self, client_results: List[Tuple[str, Any]]):
        """Return clusters based on label distribution similarity"""
        start_time = time.time()
        
        # Analyze client label distributions if not done yet
        self.analyze_client_label_distributions(client_results)
        
        # Return label-based clusters
        clusters = {}
        for cluster_name, client_list in self.label_clusters.items():
            if client_list:  # Only include non-empty clusters
                clusters[cluster_name] = client_list
        
        # Return cluster assignments
        assignments = self.cluster_assignments.copy()
        clustering_time = time.time() - start_time
        
        return clusters, assignments, clustering_time
    

    



class LeafFedAvg(FedAvg):
    def __init__(
        self,
        server_id: int,
        num_rounds: int,
        fraction_fit: float,
        fraction_evaluate: float,
        clients_per_server: int,
        initial_parameters,
        global_round: int = 0,
    ):
        # Force garbage collection before initialization
        gc.collect()
        
        # Pass most args to FedAvg base class
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=clients_per_server,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=sum_metrics,
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )
        self.server_id = server_id
        self.num_rounds = num_rounds
        self.server_str = f"Leaf Server {server_id}"
        self.global_round = global_round
        self.clients_per_server = clients_per_server
        # Get configuration from pyproject.toml
        project_root = Path(__file__).resolve().parent.parent
        cfg = toml.load(project_root / "pyproject.toml")
        hierarchy_config = cfg["tool"]["flwr"]["hierarchy"]  # ← moved up before use
        self.num_servers = hierarchy_config["num_servers"]
        
        # Set base_dir for metrics path
        self.base_dir = project_root
        
        # Multi-level clustering configuration
        cluster_config = cfg["tool"]["flwr"]["cluster"]
        multi_level_config = cluster_config.get("multi_level", {})
        self.enable_multi_level = multi_level_config.get("enable_multi_level_clustering", False)
        
        # Initialize multi-level components if enabled
        if self.enable_multi_level:
            self.metrics_tracker = MultiLevelMetrics(server_id)
            # Pass full hierarchy config for batch size access
            full_config = {**multi_level_config, **hierarchy_config}
            self.cluster_manager = ClientClusterManager(server_id, full_config)
            logger.info(f"[{self.server_str}] Multi-level clustering enabled")
        else:
            self.metrics_tracker = None
            self.cluster_manager = None
            logger.info(f"[{self.server_str}] Multi-level clustering disabled")
        
        # Communication and memory settings
        comm_config = cfg["tool"]["flwr"].get("communication", {})
        memory_config = cfg["tool"]["flwr"].get("memory", {})
        research_config = cfg["tool"]["flwr"].get("research", {})
        
        self.metrics_save_frequency = comm_config.get("metrics_save_frequency", 10)
        self.enable_gc = memory_config.get("enable_garbage_collection", True)
        self.gc_frequency = memory_config.get("gc_frequency", 5)
        self.verbose_logging = research_config.get("verbose_logging", True)
        
        # Evaluation and clustering settings — all mandatory (no fallbacks)
        try:
            self.eval_batch_size = hierarchy_config["eval_batch_size"]
            self.estimation_batch_size = hierarchy_config["estimation_batch_size"]
            self.cluster_better_delta = hierarchy_config["cluster_better_delta"]
        except KeyError as missing:
            # Fail fast and surface a clear error if any required key is missing
            raise KeyError(
                f"Required key '{missing.args[0]}' missing in [tool.flwr.hierarchy] section of TOML config."
            ) from None
        
        # Use optimized storage structure
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        self.project_root = project_root
        
        # Metrics storage removed - using direct CSV writing instead
        # Prepare validation loader once for the accuracy gate (CIFAR-10 only)
        try:
            # Use CIFAR-10 dedicated test loader for global evaluation
            self._valloader_gate = get_cifar10_test_loader(batch_size=self.eval_batch_size)
        except Exception as e:
            logger.error(f"[{self.server_str}] Error loading validation data: {e}")
            raise
        gc.collect()

    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Weighted average of accuracy metrics
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        total_examples = sum([num_examples for num_examples, _ in metrics])
        return {"accuracy": sum(accuracies) / total_examples}
    
    def _write_fit_metrics_csv(self, rnd: int, results):
        """Write client fit metrics to CSV file"""
        try:
            local_rnd = rnd - 1
            metrics_dir = self.base_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            clients_csv = metrics_dir / f"server_{self.server_id}_client_fit_metrics.csv"
            write_header = not clients_csv.exists()
            
            with open(clients_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if write_header:
                    writer.writerow([
                        'global_round', 'local_round', 'client_id', 'train_loss', 'num_examples'
                    ])
                
                for cid, fit_res in results:
                    client_id = fit_res.metrics.get("client_id", str(cid))
                    train_loss = fit_res.metrics.get("train_loss", 0.0)
                    
                    writer.writerow([
                        rnd, 0, client_id, train_loss, fit_res.num_examples
                    ])
                    
        except Exception as e:
            logger.error(f"[{self.server_str}] Failed to write fit metrics CSV: {e}")
    
    def _write_communication_metrics_csv(self, rnd: int, results):
        """Write detailed client communication metrics to CSV file"""
        try:
            metrics_dir = self.base_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            comm_csv_path = metrics_dir / f"server_{self.server_id}_client_communication_metrics.csv"
            write_header = not comm_csv_path.exists()
            
            with open(comm_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if write_header:
                    writer.writerow([
                        'global_round', 'client_id', 'bytes_up', 'bytes_down', 
                        'round_time', 'compute_time', 'bytes_transferred_total'
                    ])
                
                for cid, fit_res in results:
                    client_id = fit_res.metrics.get("client_id", str(cid))
                    bytes_up = fit_res.metrics.get("bytes_up", 0)
                    bytes_down = fit_res.metrics.get("bytes_down", 0)
                    round_time = fit_res.metrics.get("round_time", 0.0)
                    compute_time = fit_res.metrics.get("compute_s", 0.0)
                    bytes_total = bytes_up + bytes_down
                    
                    writer.writerow([
                        rnd, client_id, bytes_up, bytes_down, 
                        round_time, compute_time, bytes_total
                    ])
                    
        except Exception as e:
            logger.error(f"[{self.server_str}] Failed to write communication metrics CSV: {e}")
    
    def aggregate_evaluate(self, rnd: int, results, failures):
        """Aggregate evaluation results from clients and save metrics"""
        try:
            # Save client evaluation metrics
            self._write_eval_metrics_csv(rnd, results)
            
            # Call parent aggregate_evaluate if it exists
            if hasattr(super(), 'aggregate_evaluate'):
                return super().aggregate_evaluate(rnd, results, failures)
            else:
                # Default behavior: return weighted average of accuracy
                if not results:
                    return None
                    
                # Calculate weighted average accuracy
                total_examples = sum(num_examples for num_examples, _ in results)
                if total_examples == 0:
                    return None
                    
                weighted_accuracy = sum(
                    num_examples * metrics.get("accuracy", 0.0) 
                    for num_examples, metrics in results
                ) / total_examples
                
                return weighted_accuracy, {"accuracy": weighted_accuracy}
                
        except Exception as e:
            logger.error(f"[{self.server_str}] Error in aggregate_evaluate: {e}")
            return None
    
    def _write_eval_metrics_csv(self, rnd: int, results):
        """Write client evaluation metrics to CSV file"""
        try:
            eval_csv_path = self.base_dir / "client_eval_metrics.csv"
            write_header = not eval_csv_path.exists()
            
            with open(eval_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if write_header:
                    writer.writerow([
                        'global_round', 'client_id', 'accuracy', 'loss', 'num_examples', 
                        'bytes_down_eval'
                    ])
                
                for num_examples, metrics in results:
                    client_id = metrics.get("client_id", "unknown")
                    accuracy = metrics.get("accuracy", 0.0)
                    # Loss is typically the return value from client.evaluate()
                    loss = 0.0  # We don't have loss in the metrics, would need to modify client
                    bytes_down_eval = metrics.get("bytes_down_eval", 0)
                    
                    writer.writerow([
                        rnd, client_id, accuracy, loss, num_examples, bytes_down_eval
                    ])
                    
        except Exception as e:
            logger.error(f"[{self.server_str}] Failed to write eval metrics CSV: {e}")
    
    def aggregate_fit(self, rnd: int, results, failures):
        """Enhanced aggregation with multi-level client clustering"""
        round_start_time = time.time()
        
        # Log client results
        print(f"[{self.server_str}] Aggregating fit results from clients for round {rnd}")
        for cid, fit_res in results:
            friendly = fit_res.metrics.get("client_id", str(cid))
            loss = fit_res.metrics.get("train_loss", None)
            n = fit_res.num_examples
            print(f"  → Client {friendly} train_loss={loss:.4f} on {n} samples")
        if failures:
            for failure in failures:
                print(f"  !! Failure: {failure}")
        
        # Save client fit metrics
        self._write_fit_metrics_csv(rnd, results)
        self._write_communication_metrics_csv(rnd, results)
        
        # Memory management
        if self.enable_gc and rnd % self.gc_frequency == 0:
            gc.collect()
            if self.metrics_tracker:
                self.metrics_tracker.log_memory_usage()
        
        # Convert results to list format for processing with consistent client IDs
        client_results = []
        for idx, (cid, fit_res) in enumerate(results):
            # Create consistent client ID based on position in results list
            # This ensures deterministic clustering regardless of proxy object addresses
            consistent_client_id = f"client_{idx}"
            client_results.append((consistent_client_id, fit_res))
            logger.debug(f"[{self.server_str}] Mapped {str(cid)[:50]}... to {consistent_client_id}")
        
        # Consolidated client connection summary
        total_samples = sum(fit_res.num_examples for _, fit_res in client_results)
        logger.info(f"[{self.server_str} | Round {rnd}] Processing {len(client_results)} clients ({total_samples} total samples)")
        
        # Multi-level clustering if enabled
        if self.enable_multi_level and self.cluster_manager and len(client_results) > 0:
            aggregated = self._aggregate_with_clustering(rnd, client_results, round_start_time)
        else:
            aggregated = self._aggregate_standard(rnd, client_results, round_start_time)
        
        # Update latest parameters
        if aggregated is not None and isinstance(aggregated, tuple) and len(aggregated) > 0:
            self.latest_parameters = aggregated[0]
        
        # Save final model if last round and aggregation was successful
        if rnd == self.num_rounds and aggregated is not None:
            # Calculate total number of training examples from this round's results
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            print(f"[{self.server_str}] Total training examples: {total_examples}")

            # Extract parameters from the returned tuple
            if isinstance(aggregated, tuple) and len(aggregated) > 0:
                parameters = aggregated[0]
            else:
                # Fallback if the return type is not a tuple
                parameters = aggregated

            # Only save if parameters are valid
            if parameters is not None:
                # Save model using optimized storage
                from fedge.utils.fs_optimized import get_model_path
                model_path = get_model_path(self.project_root, self.server_id, self.global_round)

                try:
                    with open(model_path, "wb") as f:
                        pickle.dump((parameters_to_ndarrays(parameters), total_examples), f)
                    print(f"[{self.server_str}] Saved final model and sample count to {model_path}")
                except Exception as e:
                    print(f"[{self.server_str}] Failed to save model: {e}")
            else:
                print(f"[{self.server_str}] Cannot save model: parameters are None")
        elif rnd == self.num_rounds and aggregated is None:
            print(f"[{self.server_str}] Cannot save model: aggregation returned None (no successful client updates)")
        
        return aggregated
    
    def _aggregate_with_clustering(self, rnd: int, client_results, round_start_time):
        """Perform multi-level aggregation with client clustering"""
        try:
            # Step 1: Cluster clients by label distribution similarity
            clusters, assignments, clustering_time = self.cluster_manager.cluster_clients_by_characteristics(client_results)
            
            if self.verbose_logging:
                logger.info(f"[{self.server_str} | Round {rnd}] Client clustering completed in {clustering_time:.3f}s")
                for cluster_name, client_list in clusters.items():
                    logger.info(f"  - {cluster_name}: {len(client_list)} clients {client_list}")
            
            # Step 2: Train cluster-specific models
            cluster_models, training_times = self._train_cluster_models(clusters, client_results, self.initial_parameters)
            
            # Step 3: Aggregate cluster models into server model
            server_model = self._create_server_model(cluster_models, clusters)
            
            # Step 4: Calculate cluster accuracies
            cluster_accuracies = self._calculate_cluster_accuracies(clusters, client_results)
            
            # Step 5: Evaluate server model
            server_accuracy = self._evaluate_server_model(server_model)
            
            # Step 6: Estimate communication overhead
            bytes_transferred = self._estimate_bytes_transferred(client_results)
        
            
            # Log metrics
            if self.metrics_tracker:
                aggregation_time = time.time() - round_start_time - clustering_time
                self.metrics_tracker.log_computation(clustering_time, aggregation_time)
                
                cluster_sizes = {name: len(clients) for name, clients in clusters.items()}
                self.metrics_tracker.log_performance(cluster_accuracies, cluster_sizes, server_accuracy)
                
                round_time = time.time() - round_start_time
                self.metrics_tracker.log_communication(round_time, len(client_results), bytes_transferred)
                
                # Metrics tracking removed - using direct CSV writing instead
            
            if self.verbose_logging:
                logger.info(f"[{self.server_str} | Round {rnd}] Multi-level aggregation completed:")
                logger.info(f"  - Clusters: {len(clusters)}")
                logger.info(f"  - Server accuracy: {server_accuracy:.4f}")
                logger.info(f"  - Cluster accuracies: {cluster_accuracies}")
            
            # Update parameters
            self.parameters = server_model
            
            # Return in FedAvg format
            return server_model, {}
            
        except Exception as e:
            logger.error(f"[{self.server_str} | Round {rnd}] Clustering aggregation failed: {e}")
            logger.info(f"[{self.server_str} | Round {rnd}] Falling back to standard aggregation")
            return self._aggregate_standard(rnd, client_results, round_start_time)
    
    def _aggregate_standard(self, rnd: int, client_results, round_start_time):
        """Standard FedAvg aggregation without clustering"""
        if self.verbose_logging:
            logger.info(f"[{self.server_str} | Round {rnd}] Using standard FedAvg aggregation")
        
        # Check if we have any client results
        if not client_results:
            logger.warning(f"[{self.server_str} | Round {rnd}] No client results to aggregate")
            return None
        
        # Call parent FedAvg aggregation
        aggregated = super().aggregate_fit(rnd, client_results, [])
        
        # Check if aggregation was successful
        if aggregated is None:
            logger.warning(f"[{self.server_str} | Round {rnd}] Parent FedAvg aggregation returned None")
        
        # Log metrics
        if self.metrics_tracker:
            round_time = time.time() - round_start_time
            bytes_transferred = self._estimate_bytes_transferred(client_results)
            self.metrics_tracker.log_communication(round_time, len(client_results), bytes_transferred)
        
        return aggregated
    
    def _train_cluster_models(self, clusters: dict, client_results, global_model):
        """Train separate models for each cluster using weighted averaging"""
        cluster_models = {}
        training_times = {}
        
        result_dict = {str(cid): fit_res for cid, fit_res in client_results}
        
        for cluster_name, client_ids in clusters.items():
            start_time = time.time()
            
            # Get client results for this cluster
            cluster_client_results = []
            for client_id in client_ids:
                if client_id in result_dict:
                    cluster_client_results.append((client_id, result_dict[client_id]))
            
            if cluster_client_results:
                # Weighted average within cluster
                total_examples = sum(fit_res.num_examples for _, fit_res in cluster_client_results)
                
                if total_examples > 0:
                    aggregated_params = None
                    
                    for client_id, fit_res in cluster_client_results:
                        weight = fit_res.num_examples / total_examples
                        client_arrays = parameters_to_ndarrays(fit_res.parameters)
                        
                        if aggregated_params is None:
                            aggregated_params = [weight * param for param in client_arrays]
                        else:
                            for i, param in enumerate(client_arrays):
                                aggregated_params[i] += weight * param
                    
                    cluster_models[cluster_name] = ndarrays_to_parameters(aggregated_params)
                else:
                    cluster_models[cluster_name] = global_model
            else:
                cluster_models[cluster_name] = global_model
            
            training_times[cluster_name] = time.time() - start_time
        
        return cluster_models, training_times
    
    def _create_server_model(self, cluster_models: dict, clusters: dict):
        """Create server model by aggregating cluster models weighted by cluster size"""
        if not cluster_models:
            return self.parameters
        
        # Calculate weights based on cluster sizes
        total_clients = sum(len(clients) for clients in clusters.values())
        if total_clients == 0:
            return self.parameters
        
        cluster_weights = {}
        for cluster_name, clients in clusters.items():
            cluster_weights[cluster_name] = len(clients) / total_clients
        
        # Aggregate cluster models
        aggregated_params = None
        
        for cluster_name, model_params in cluster_models.items():
            weight = cluster_weights.get(cluster_name, 0)
            if weight > 0:
                cluster_arrays = parameters_to_ndarrays(model_params)
                
                if aggregated_params is None:
                    aggregated_params = [weight * param for param in cluster_arrays]
                else:
                    for i, param in enumerate(cluster_arrays):
                        aggregated_params[i] += weight * param
        
        if aggregated_params is None:
            return self.parameters
        
        return ndarrays_to_parameters(aggregated_params)
    
    def _calculate_cluster_accuracies(self, clusters: dict, client_results):
        """Calculate average accuracy for each cluster"""
        cluster_accuracies = {}
        result_dict = {str(cid): fit_res for cid, fit_res in client_results}
        
        for cluster_name, client_ids in clusters.items():
            accuracies = []
            for client_id in client_ids:
                if client_id in result_dict:
                    fit_res = result_dict[client_id]
                    if hasattr(fit_res, 'metrics') and 'accuracy' in fit_res.metrics:
                        accuracies.append(fit_res.metrics['accuracy'])
            
            if accuracies:
                cluster_accuracies[cluster_name] = np.mean(accuracies)
            else:
                cluster_accuracies[cluster_name] = 0.0
        
        return cluster_accuracies
    
    def _evaluate_server_model(self, server_model):
        """Evaluate server model on participating clients' validation data"""
        try:
            # Get device for evaluation
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # Create a temporary model with server parameters
            temp_model = Net()
            set_weights(temp_model, parameters_to_ndarrays(server_model))
            
            # Evaluate on union of participating clients' validation data
            total_examples = 0
            total_loss = 0.0
            total_correct = 0
            
            for cid_local in range(self.clients_per_server):
                # Use local client ID as partition ID (each server has its own partition namespace)
                part_id = cid_local
                
                # Load validation data for this client
                dataset_flag = "cifar10"
                _, valloader, _ = load_data(dataset_flag, part_id, self.num_servers * self.clients_per_server, batch_size=self.eval_batch_size, server_id=self.server_id)
                
                # Evaluate on this client's validation data
                loss, accuracy = test(temp_model, valloader, device)
                n_examples = len(valloader.dataset)
                
                total_examples += n_examples
                total_loss += loss * n_examples
                total_correct += accuracy * n_examples
            
            # Return weighted average accuracy across participating clients
            if total_examples > 0:
                return total_correct / total_examples
            else:
                return 0.0
                
        except Exception as e:
            # Strict policy: no fallback to server validation gate
            logger.warning(f"[{self.server_str}] Could not evaluate server model (no fallback): {e}")
            return 0.0
    
    def _estimate_bytes_transferred(self, client_results):
        """Estimate bytes transferred based on model parameters"""
        if not client_results:
            return 0
        
        # Rough estimate: assume each parameter is 4 bytes (float32)
        try:
            sample_result = client_results[0][1]
            if hasattr(sample_result, 'parameters'):
                param_arrays = parameters_to_ndarrays(sample_result.parameters)
                total_params = sum(arr.size for arr in param_arrays)
                bytes_per_client = total_params * 4  # 4 bytes per float32
                return bytes_per_client * len(client_results)
        except Exception:
            pass
        
        return 1024 * len(client_results)  # Fallback estimate

    def _write_fit_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        # Save per-client fit metrics into strict round folder
        local_rnd = rnd - 1
        clients_csv = self.base_dir / "client_fit_metrics.csv"
        write_header = not clients_csv.exists()
        with open(clients_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["global_round","local_round","client_id","train_loss","num_examples"])
            if write_header:
                writer.writeheader()

            for cid, fit_res in results:
                cid_str = fit_res.metrics.get("client_id", str(cid))
                loss = fit_res.metrics.get("train_loss")
                n = fit_res.num_examples
                writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "client_id": cid_str,
                    "train_loss": loss,
                    "num_examples": n,
                })
        logger.info(f"[{self.server_str} | Round {local_rnd}] Appended client fit metrics → {clients_csv}")

    def _write_eval_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        """Save per-client evaluation loss & accuracy for this round"""
        local_rnd = rnd - 1
        metrics_dir = self.base_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        clients_eval_csv = metrics_dir / f"server_{self.server_id}_client_eval_metrics.csv"
        write_header = not clients_eval_csv.exists()
        with open(clients_eval_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["global_round","local_round","client_id","eval_loss","accuracy","num_examples"])
            if write_header:
                writer.writeheader()
            for cid, eval_res in results:
                cid_str = eval_res.metrics.get("client_id", str(cid))
                loss = eval_res.loss
                acc = eval_res.metrics.get("accuracy")
                n = eval_res.num_examples
                writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "client_id": cid_str,
                    "eval_loss": loss,
                    "accuracy": acc,
                    "num_examples": n,
                })
        logger.info(f"[{self.server_str} | Round {local_rnd}] Appended client eval metrics → {clients_eval_csv}")

    # ------------------------------------------------------------------
    #  Optional accuracy gate: Only accept incoming global/cluster model
    #  if it improves validation accuracy on this leaf's data.
    # ------------------------------------------------------------------
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # ✅ CRITICAL FIX: Get base fit configurations FIRST
        fit_cfgs = super().configure_fit(server_round, parameters, client_manager)
        
        # ✅ CRITICAL FIX: Set all client configs BEFORE any early returns
        # This ensures SCAFFOLD and FedProx are NEVER skipped by the accuracy gate
        
        # Get SCAFFOLD setting from TOML config only (no environment overrides)
        scaffold_enabled = False  # Will be set from TOML config below
        
        # ✅ NEW: Load ALL client hyperparameters from TOML config
        try:
            import toml
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent
            with open(project_root / "pyproject.toml", "r", encoding="utf-8") as f:
                config = toml.load(f)
            app_cfg = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
            hierarchy_cfg = config.get("tool", {}).get("flwr", {}).get("hierarchy", {})
        except Exception as e:
            logger.warning(f"Could not load TOML config: {e}, using defaults")
            app_cfg = {}
            hierarchy_cfg = {}
        # Get scaffold_enabled from TOML config only
        if "scaffold_enabled" not in hierarchy_cfg:
            raise ValueError("scaffold_enabled must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
        scaffold_enabled = bool(hierarchy_cfg["scaffold_enabled"])
        
        # ✅ CRITICAL: Extract client hyperparameters from environment
        # Use LR_INIT from environment as single source of truth for learning rate
        lr_init_env = os.environ.get("LR_INIT")
        if lr_init_env is None:
            raise ValueError("LR_INIT environment variable is required but not set")
        client_lr = float(lr_init_env)
        
        required_hierarchy_params = ['weight_decay', 'clip_norm', 'momentum', 'lr_gamma']
        for param in required_hierarchy_params:
            if param not in hierarchy_cfg:
                raise ValueError(f"{param} must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
        
        weight_decay = float(hierarchy_cfg["weight_decay"])
        clip_norm = float(hierarchy_cfg["clip_norm"])
        momentum = float(hierarchy_cfg["momentum"])
        lr_gamma = float(hierarchy_cfg["lr_gamma"])
        
        # Get FedProx parameters from TOML config only
        if "prox_mu" not in hierarchy_cfg:
            raise ValueError("prox_mu must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
        mu_base = float(hierarchy_cfg["prox_mu"])
        
        # These FedProx adaptive parameters should also come from TOML
        required_hierarchy_params = ['prox_mu_min', 'acc_stable_threshold', 'loss_delta']
        for param in required_hierarchy_params:
            if param not in hierarchy_cfg:
                raise ValueError(f"{param} must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
        
        mu_min = float(hierarchy_cfg["prox_mu_min"])
        acc_stable = float(hierarchy_cfg["acc_stable_threshold"])
        loss_delta = float(hierarchy_cfg["loss_delta"])
        
        # Load per-client eval metrics for adaptive FedProx
        import csv
        metrics_path = self.base_dir / "client_eval_metrics.csv"
        rows = list(csv.DictReader(open(metrics_path))) if metrics_path.exists() else []
        
        # ✅ CRITICAL: Configure ALL clients with SCAFFOLD + FedProx + Hyperparams BEFORE gate check
        for cid, fit_ins in fit_cfgs:
            # ✅ FIX #1: Pass SCAFFOLD through config instead of environment
            fit_ins.config["scaffold_enabled"] = scaffold_enabled
            
            # ✅ NEW: Pass ALL client hyperparameters through config
            fit_ins.config["learning_rate"] = client_lr
            fit_ins.config["weight_decay"] = weight_decay
            fit_ins.config["clip_norm"] = clip_norm
            fit_ins.config["momentum"] = momentum
            fit_ins.config["lr_gamma"] = lr_gamma
            
            # ✅ FIX #2: Calculate adaptive FedProx proximal_mu per client
            cid_str = fit_ins.config.get("client_id", str(cid))
            mu = mu_base
            if rows and self.global_round >= 1:
                last_round = self.global_round - 1
                prev_round = self.global_round - 2
                client_rows = [r for r in rows if r.get("client_id") == cid_str]
                last_entries = [r for r in client_rows if int(r.get("global_round", -1)) == last_round]
                prev_entries = [r for r in client_rows if int(r.get("global_round", -1)) == prev_round]
                if last_entries and prev_entries:
                    best_last = max(last_entries, key=lambda r: int(r.get("local_round", 0)))
                    best_prev = max(prev_entries, key=lambda r: int(r.get("local_round", 0)))
                    acc_last = float(best_last.get("accuracy", 0.0))
                    loss_last = float(best_last.get("eval_loss", 0.0))
                    loss_prev = float(best_prev.get("eval_loss", 0.0))
                    # If client stable (high acc and loss not decreasing), reduce μ
                    if acc_last >= acc_stable and loss_last >= loss_prev - loss_delta:
                        mu = mu_min
            fit_ins.config["proximal_mu"] = mu
            fit_ins.config["global_round"] = self.global_round
            
            # ✅ SCAFFOLD: Pass server control variates from cloud if available
            # This comes from the cloud server's configure_fit method
            if "scaffold_server_control" in fit_ins.config:
                # Server control variates are already serialized by cloud server
                logger.debug(f"[{self.server_str}] SCAFFOLD server control variates received from cloud")
        
        # ✅ NOW apply accuracy gate check (AFTER configs are safely set)
        gate = os.getenv("CLUSTER_REPLACE_IF_BETTER", "1") != "0"
        delta = self.cluster_better_delta  # From TOML config — mandatory, no default
        if gate and parameters is not None and hasattr(self, "latest_parameters") and self.latest_parameters is not None:
            try:
                dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # Evaluate old params
                model_old = Net().to(dev)
                set_weights(model_old, parameters_to_ndarrays(self.latest_parameters))
                _, acc_old = test(model_old, self._valloader_gate, dev)
                # Evaluate new params
                model_new = Net().to(dev)
                set_weights(model_new, parameters_to_ndarrays(parameters))
                _, acc_new = test(model_new, self._valloader_gate, dev)
                if acc_new < acc_old + delta:
                    # Keep old parameters but preserve our custom configs
                    logger.info(f"[{self.server_str}] Accuracy gate: keeping old params (old={acc_old:.4f}, new={acc_new:.4f})")
                    parameters = self.latest_parameters
                    # Re-create fit_cfgs with old parameters but keep our configs
                    old_fit_cfgs = super().configure_fit(server_round, parameters, client_manager)
                    # Preserve our custom SCAFFOLD + FedProx configs
                    for i, (old_cid, old_fit_ins) in enumerate(old_fit_cfgs):
                        if i < len(fit_cfgs):
                            _, original_fit_ins = fit_cfgs[i]
                            old_fit_ins.config.update(original_fit_ins.config)
                    fit_cfgs = old_fit_cfgs
                else:
                    logger.info(f"[{self.server_str}] Accuracy gate: using new params (old={acc_old:.4f}, new={acc_new:.4f})")
            except Exception as e:
                logger.error(f"[{self.server_str}] Accuracy gate error: {e}")
        
        return fit_cfgs



    def aggregate_evaluate(self, rnd, results, failures):
        # 1) Save per-client evaluation metrics
        self._write_eval_metrics_csv(rnd, results)

        # 2) Aggregate client evaluation (FedAvg returns (loss, metrics))
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        if aggregated is not None and isinstance(aggregated, tuple):
            agg_loss = aggregated[0]
            agg_acc = aggregated[1].get("accuracy") if aggregated[1] else None
        else:
            agg_loss, agg_acc = None, None

        # 3) Centralized evaluation on full *global* test set
        local_rnd = rnd - 1
        model = Net()
        if hasattr(self, "latest_parameters") and self.latest_parameters is not None:
            nds = parameters_to_ndarrays(self.latest_parameters)
            set_weights(model, nds)
        dev = torch.device("cpu")

        # 3-a Global test set (shared by all servers) - use dedicated CIFAR-10 test loader
        from fedge.task import get_cifar10_test_loader
        batch_size = getattr(self, 'eval_batch_size', 20)  # From TOML config
        testloader_global = get_cifar10_test_loader(batch_size=batch_size)
        central_loss, central_acc = test(model, testloader_global, dev)

        # 3-b Local-centralised test set (union of this server's clients)
        dataset_flag = "cifar10"  # Fixed: define dataset_flag
        total_clients = self.num_servers * self.clients_per_server
        local_examples = 0
        local_loss_sum = 0.0
        local_acc_sum = 0.0
        for cid_local in range(self.clients_per_server):
            # Use cid_local directly as partition_id since each server has its own partition namespace
            part_id = cid_local
            _, testloader_local, _ = load_data(dataset_flag, part_id, total_clients, batch_size=batch_size, server_id=self.server_id)
            l_loss, l_acc = test(model, testloader_local, dev)
            n = len(testloader_local.dataset)
            local_examples += n
            local_loss_sum += l_loss * n
            local_acc_sum += l_acc * n
        local_loss = local_loss_sum / local_examples if local_examples else None
        local_acc = local_acc_sum / local_examples if local_examples else None

        # 4) Compute gaps and statistical metrics
        gap_global_loss = central_loss - (agg_loss or 0.0)
        gap_global_acc = central_acc - (agg_acc or 0.0)
        gap_local_loss = local_loss - (agg_loss or 0.0) if local_loss is not None else None
        gap_local_acc = local_acc - (agg_acc or 0.0) if local_acc is not None else None

        # 4-b) Calculate standard deviation and confidence intervals from client results
        import numpy as np
        from scipy import stats
        
        client_losses = [res[1].loss for res in results if res[1].loss is not None]
        client_accs = [res[1].metrics.get("accuracy", 0) for res in results if res[1].metrics.get("accuracy") is not None]
        
        # Standard deviations
        loss_std = np.std(client_losses) if len(client_losses) > 1 else 0.0
        acc_std = np.std(client_accs) if len(client_accs) > 1 else 0.0
        
        # 95% Confidence intervals (assuming normal distribution)
        loss_ci_lower, loss_ci_upper = None, None
        acc_ci_lower, acc_ci_upper = None, None
        
        if len(client_losses) > 1:
            loss_mean = np.mean(client_losses)
            loss_sem = stats.sem(client_losses)  # Standard error of mean
            loss_ci = stats.t.interval(0.95, len(client_losses)-1, loc=loss_mean, scale=loss_sem)
            loss_ci_lower, loss_ci_upper = loss_ci
            
        if len(client_accs) > 1:
            acc_mean = np.mean(client_accs)
            acc_sem = stats.sem(client_accs)
            acc_ci = stats.t.interval(0.95, len(client_accs)-1, loc=acc_mean, scale=acc_sem)
            acc_ci_lower, acc_ci_upper = acc_ci

        # 5) Create metrics folder and write server-side metrics CSV
        metrics_dir = self.base_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        server_csv = metrics_dir / f"server_{self.server_id}_metrics.csv"
        write_header = not server_csv.exists()
        with open(server_csv, "a", newline="") as fsv:
            writer = csv.DictWriter(
                fsv,
                fieldnames=[
                    "global_round","local_round",
                    "agg_loss","agg_acc",
                    "central_loss","central_acc",
                    "local_loss","local_acc",
                    "gap_global_loss","gap_global_acc",
                    "gap_local_loss","gap_local_acc",
                    "loss_std","acc_std",
                    "loss_ci_lower","loss_ci_upper",
                    "acc_ci_lower","acc_ci_upper",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": self.global_round,
                "local_round": local_rnd,
                "agg_loss": agg_loss,
                "agg_acc": agg_acc,
                "central_loss": central_loss,
                "central_acc": central_acc,
                "local_loss": local_loss,
                "local_acc": local_acc,
                "gap_global_loss": gap_global_loss,
                "gap_global_acc": gap_global_acc,
                "gap_local_loss": gap_local_loss,
                "gap_local_acc": gap_local_acc,
                "loss_std": loss_std,
                "acc_std": acc_std,
                "loss_ci_lower": loss_ci_lower,
                "loss_ci_upper": loss_ci_upper,
                "acc_ci_lower": acc_ci_lower,
                "acc_ci_upper": acc_ci_upper,
            })
        logger.info(f"[{self.server_str} | Round {local_rnd}] Appended server metrics → {server_csv}")
        return aggregated


def handle_signal(sig, frame):
    """Handle termination signals gracefully"""
    server_str = os.environ.get("SERVER_STR", "Leaf Server")
    logger.info(f"[{server_str}] Received signal {sig}, shutting down gracefully...")
    
    # If we know the server ID, try to save the model before exiting
    server_id = os.environ.get("SERVER_ID")
    if server_id:
        try:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            model_dir = project_root / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"server_{server_id}.pkl"
            
            # Try to get a model instance
            net = Net()
            ndarrays = get_weights(net)
            
            # Save with a default example count
            with open(model_path, "wb") as f:
                pickle.dump((ndarrays, 0), f)
            logger.info(f"[{server_str}] Saved emergency backup model during shutdown to {model_path}")
        except Exception as e:
            logger.error(f"[{server_str}] Failed to save emergency model: {e}")
    
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--clients_per_server", type=int, required=True)
    parser.add_argument("--num_rounds", type=int, required=True)
    parser.add_argument("--fraction_fit", type=float, required=True)
    parser.add_argument("--fraction_evaluate", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--initial_model_path", type=str, help="Path to global model to start from")
    parser.add_argument("--global_round", type=int, default=0, help="Current global round number (0-indexed)")
    parser.add_argument("--start_round", type=int, default=0, help="Starting round number (0-indexed)")
    parser.add_argument("--dir_round", type=int, help="Round number for directory naming (0-indexed)")
    
    args = parser.parse_args()

    # Get server ID for logging
    server_id = args.server_id
    server_str = f"Leaf Server {server_id}"
    
    # Store in environment for signal handlers
    os.environ["SERVER_ID"] = str(server_id)
    os.environ["SERVER_STR"] = server_str
    
    logger.info(f"[{server_str}] Starting server with {args.clients_per_server} clients, {args.num_rounds} rounds")
    
    # Initialize model and get parameters
    initial_parameters = None
    if args.initial_model_path:
        model_path = Path(args.initial_model_path)
        if model_path.exists():
            logger.info(f"[{server_str}] Loading initial model from {model_path}")
            try:
                with open(model_path, "rb") as f:
                    # The global model could be stored in different formats
                    loaded_data = pickle.load(f)
                    if isinstance(loaded_data, tuple):
                        # If it's a tuple, first element should be the model parameters
                        ndarrays = loaded_data[0]
                    else:
                        # Otherwise, assume it's the raw parameters
                        ndarrays = loaded_data
                logger.info(f"[{server_str}] Starting with global model from round {args.global_round}")
                initial_parameters = ndarrays_to_parameters(ndarrays)
            except Exception as e:
                logger.error(f"[{server_str}] Error loading initial model: {e}")
                logger.info(f"[{server_str}] Starting with fresh model parameters")
    
    # Configure strategy
    strategy = LeafFedAvg(
        server_id=server_id,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        clients_per_server=args.clients_per_server, 
        initial_parameters=initial_parameters,
        global_round=args.global_round,
    )
    
    # Start server with error handling
    logger.info(f"[{server_str}] Starting Flower server on port {args.port}")
    try:
        bind_addr = os.getenv("BIND_ADDRESS", "0.0.0.0")
        # Run blocking Flower server and capture History object for metrics
        # Temporarily redirect stderr to suppress deprecation warnings
        import contextlib
        import io
        
        stderr_capture = io.StringIO()
        with contextlib.redirect_stderr(stderr_capture):
            history = start_server(
                server_address=f"{bind_addr}:{args.port}",
                config=ServerConfig(num_rounds=args.num_rounds),
                strategy=strategy,
            )
        logger.info(f"[{server_str}] Server has completed all rounds successfully")
        
        # Export final server metrics to CSV
        try:
            if strategy.enable_multi_level and strategy.metrics_tracker:
                # Export server-level metrics for orchestrator
                server_metrics_path = strategy.base_dir / "server_metrics.csv"
                
                # Calculate final server metrics
                if strategy.metrics_tracker.performance_metrics:
                    latest_performance = strategy.metrics_tracker.performance_metrics[-1]
                    server_accuracy = latest_performance.get('server_accuracy', 0.0)
                    total_clients = sum(latest_performance.get('cluster_sizes', {}).values())
                else:
                    server_accuracy = 0.0
                    total_clients = args.clients_per_server
                
                # Do NOT overwrite per-round metrics with placeholder values.
                # Preserve existing metrics written during training; if file doesn't exist,
                # initialize only the header (no placeholder rows).
                try:
                    if server_metrics_path.exists():
                        logger.info(f"[{server_str}] Preserved existing server metrics at {server_metrics_path}")
                    else:
                        with open(server_metrics_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['global_round', 'local_round', 'agg_loss', 'agg_acc', 'central_loss', 'central_acc', 'local_loss', 'local_acc', 'gap_global_loss', 'gap_global_acc', 'gap_local_loss', 'gap_local_acc'])
                        logger.info(f"[{server_str}] Initialized server metrics CSV at {server_metrics_path}")
                except Exception as init_err:
                    logger.warning(f"[{server_str}] Skipped final metrics write to avoid placeholders: {init_err}")
        except Exception as metrics_err:
            logger.warning(f"[{server_str}] Could not write server metrics CSV: {metrics_err}")
    
    except UnicodeEncodeError as e:
        logger.error(f"[{server_str}] Unicode encoding error: {str(e)}")
        logger.error(f"[{server_str}] This is likely due to special characters in logging output")
        # Continue with metrics saving
        history = None
    except Exception as e:
        import traceback
        logger.error(f"[{server_str}] Server error: {type(e).__name__}: {str(e)}")
        logger.error(f"[{server_str}] Full traceback: {traceback.format_exc()}")
        
        # Check for specific error types
        if "address already in use" in str(e).lower() or "bind" in str(e).lower():
            logger.error(f"[{server_str}] PORT CONFLICT: Another process is using port {args.port}")
        elif "connection" in str(e).lower():
            logger.error(f"[{server_str}] CONNECTION ERROR: Network connectivity issue")
        
        # Don't return here - continue with cleanup
        history = None
        
    # Always try to export metrics even if server had errors
    if history is None:
        logger.warning(f"[{server_str}] No history available, skipping communication metrics export")
        # Still try to create completion signal
        try:
            # Create completion signal using optimized storage
            global_round = int(os.getenv('GLOBAL_ROUND', '1'))
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            from fedge.utils.fs_optimized import create_completion_signal
            completion_signal = create_completion_signal(project_root, global_round, server_id)
            logger.info(f"[{server_str}] Created completion signal at {completion_signal}")
        except Exception as signal_err:
            logger.warning(f"[{server_str}] Could not create completion signal: {signal_err}")
        
        # Perform cleanup and exit gracefully
        print(f"[{server_str}] About to call os._exit(0) after cleanup")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"[{server_str}] Cleanup completed (no history), exiting gracefully")
            logging.shutdown()
        except Exception as cleanup_err:
            print(f"[{server_str}] Cleanup error: {cleanup_err}")
        
        # Use os._exit(0) because background threads prevent sys.exit(0) from working
        os._exit(0)
        return  # This should never be reached, but just in case
        
    # ------------------------------------------------------------------
    # Persist round-level communication CSV for this leaf server
    # ------------------------------------------------------------------
    try:
        import pandas as pd, filelock
        logger.info(f"[{server_str}] Starting communication metrics export")
        
        # Extract round-level communication metrics from History
        # Build per-round communication metrics from History.metrics_distributed_fit
        mdf = getattr(history, "metrics_distributed_fit", {}) or {}

        def _collect(keys):
            for k in keys:
                if k in mdf:
                    return mdf[k]
            return []

        up_entries = _collect(["bytes_up", "bytes_written"])
        down_entries = _collect(["bytes_down", "bytes_read"])
        rt_entries = _collect(["round_time"])  # mean time per round
        comp_entries = _collect(["compute_s"])  # mean compute per round

        by_round: dict[int, dict[str, float]] = {}
        for rnd, val in up_entries:
            by_round.setdefault(rnd, {})["bytes_up"] = int(val)
        for rnd, val in down_entries:
            by_round.setdefault(rnd, {})["bytes_down"] = int(val)
        for rnd, val in rt_entries:
            by_round.setdefault(rnd, {})["round_time"] = float(val)
        for rnd, val in comp_entries:
            by_round.setdefault(rnd, {})["compute_s"] = float(val)

        rows = [
            {
                "global_round": args.global_round,
                "round": rnd,
                "bytes_up": int(vals.get("bytes_up", 0)),
                "bytes_down": int(vals.get("bytes_down", 0)),
                "round_time": vals.get("round_time", 0.0),
                "compute_s": vals.get("compute_s", 0.0),
            }
            for rnd, vals in sorted(by_round.items())
        ]
        if rows:
            df = pd.DataFrame(rows)
            out = Path(os.getenv("RUN_DIR", ".")) / f"edge_comm_{server_id}.csv"
            with filelock.FileLock(out.with_suffix(".lock")):
                mode = "a" if out.exists() else "w"
                df.to_csv(out, index=False, mode=mode, header=not out.exists())
            logger.info(f"[{server_str}] Wrote communication CSV to {out}")
        else:
            logger.warning(f"[{server_str}] No communication metrics to export")
            
    except ImportError as import_err:
        logger.warning(f"[{server_str}] Could not import pandas/filelock for comm CSV: {import_err}")
    except Exception as csv_err:
        logger.warning(f"[{server_str}] Could not write comm CSV: {csv_err}")
    
    # Create completion signal to indicate server finished successfully
    try:
        # Create completion signal using optimized storage
        global_round = int(os.getenv('GLOBAL_ROUND', '1'))
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        from fedge.utils.fs_optimized import create_completion_signal
        completion_signal = create_completion_signal(project_root, global_round, server_id)
        logger.info(f"[{server_str}] Created completion signal at {completion_signal}")
    except Exception as signal_err:
        logger.warning(f"[{server_str}] Could not create completion signal: {signal_err}")
    
    logger.info(f"[{server_str}] Process completed successfully")
    
    # Perform cleanup and exit gracefully
    try:
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Shutdown logging
        logging.shutdown()
        
        logger.info(f"[{server_str}] Cleanup completed, exiting gracefully")
    except Exception as cleanup_err:
        print(f"[{server_str}] Cleanup error: {cleanup_err}")
    
    # Exit gracefully to allow proper Python shutdown sequence
    sys.exit(0)


def main_wrapper():
    """Wrapper to ensure graceful exit even on exceptions"""
    try:
        main()
    except Exception as e:
        server_str = os.environ.get("SERVER_STR", "Leaf Server")
        logger.error(f"[{server_str}] Main function error: {e}")
        # Still perform cleanup on error
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logging.shutdown()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    gc.collect()
    main_wrapper()
