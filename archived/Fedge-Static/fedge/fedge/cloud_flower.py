# cloud_flower.py
"""
Cloud server script for hierarchical federated learning using Flower ≥1.13.
This script starts a Flower ServerApp with communication-cost modifiers,
writes communication metrics to CSV after completion, and handles
dynamic clustering and signal files.
"""

# Suppress ALL warnings before any imports to catch early deprecation warnings
# COMPREHENSIVE DEPRECATION WARNING SUPPRESSION
import warnings
import sys
import os

# Multiple layers of warning suppression
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED FEATURE.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*start_server.*")
warnings.filterwarnings("ignore", module="flwr")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore")

# Set environment variable to suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# Redirect stderr during imports to catch C-level warnings
class _SuppressOutput:
    def __init__(self):
        self._original_stderr = sys.stderr
        self._original_stdout = sys.stdout
    
    def __enter__(self):
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = self._original_stderr
        sys.stdout = self._original_stdout

import os
import sys
import time
import json
import csv
import pickle
import base64
import threading
import signal
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import filelock
import toml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.server import start_server, ServerConfig
from flwr.common import Metrics, Parameters, NDArrays
# PropertiesIns may be in different modules depending on Flower version
try:
    from flwr.common import PropertiesIns
except ImportError:
    try:
        from flwr.common.protocol import PropertiesIns
    except ImportError:
        PropertiesIns = None
from flwr.server.strategy import FedAdam
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# Import advanced federated learning components
try:
    from scaffold_utils import SCAFFOLDControlVariates, create_scaffold_manager
    from dynamic_clustering import DynamicWeightClustering, create_dynamic_clustering
    from task import load_data, Net, test, get_transform, get_weights, get_cifar10_test_loader  # Net is the model class
except ImportError as e:
    logger.warning(f"Could not import advanced FL components: {e}")
    # Graceful fallbacks
    SCAFFOLDControlVariates = None
    create_scaffold_manager = None
    DynamicWeightClustering = None
    create_dynamic_clustering = None
    Net = None

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

# Static View Strategy for view-based federated learning
class StaticViewStrategy(FedAdam):
    """Keep one independent FedAdam optimizer per fixed cluster/server.
    
    This strategy maintains separate model parameters for each server/view,
    preventing cross-view contamination in view-based federated learning.
    """
    
    def __init__(self, num_heads: int, **kwargs):
        # Store num_heads before calling super()
        self.num_heads = num_heads
        
        # Initialize parent with base parameters
        super().__init__(**kwargs)
        
        # Initialize one parameter set per head (server/view)
        initial_params = kwargs.get("initial_parameters")
        if initial_params:
            self.head_params = [initial_params] * num_heads
        else:
            # Fallback to creating parameters from a model
            model = Net()
            initial_params = ndarrays_to_parameters(get_weights(model))
            self.head_params = [initial_params] * num_heads
        
        # Initialize optimizer state for each head
        self.head_opt_state = [self._init_optimizer_state() for _ in range(num_heads)]
        
        logger.info(f"[StaticViewStrategy] Initialized {num_heads} independent model heads")
        
        # Model persistence directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Try to load existing models from previous global round
        self._load_head_models()
    
    def _init_optimizer_state(self):
        """Initialize optimizer state for a single head."""
        # Initialize empty state - will be populated during training
        return {}
    
    def _load_head_models(self):
        """Load head models from disk if they exist from previous global rounds."""
        import pickle
        
        for head_id in range(self.num_heads):
            model_path = self.models_dir / f"head_{head_id}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        loaded_data = pickle.load(f)
                        if isinstance(loaded_data, tuple):
                            # Format: (ndarrays, num_examples)
                            ndarrays = loaded_data[0]
                        else:
                            # Raw ndarrays
                            ndarrays = loaded_data
                        
                        # Convert to Parameters and store
                        self.head_params[head_id] = ndarrays_to_parameters(ndarrays)
                        logger.info(f"[StaticViewStrategy] Loaded head {head_id} model from {model_path}")
                except Exception as e:
                    logger.warning(f"[StaticViewStrategy] Failed to load head {head_id} model: {e}")
                    logger.info(f"[StaticViewStrategy] Head {head_id} will start with fresh parameters")
            else:
                logger.info(f"[StaticViewStrategy] No existing model for head {head_id}, using fresh parameters")
    
    def _save_head_models(self):
        """Save all head models to disk for next global round."""
        import pickle
        
        for head_id in range(self.num_heads):
            model_path = self.models_dir / f"head_{head_id}.pkl"
            try:
                # Convert Parameters to ndarrays for saving
                ndarrays = parameters_to_ndarrays(self.head_params[head_id])
                
                # Save in format compatible with leaf server loading
                with open(model_path, "wb") as f:
                    pickle.dump((ndarrays, 0), f)  # (parameters, num_examples)
                
                logger.info(f"[StaticViewStrategy] Saved head {head_id} model to {model_path}")
            except Exception as e:
                logger.error(f"[StaticViewStrategy] Failed to save head {head_id} model: {e}")
    
    def _get_server_id(self, client_proxy) -> int | None:
        """Return the logical server_id (0, 1, 2) reported by a proxy using new Flower API."""
        try:
            # Use new Flower ≥ 1.4 API
            res = client_proxy.get_properties(PropertiesIns(config={}), timeout=30)
            props = res.properties if hasattr(res, "properties") else res  # works for old + new return types
            
            # Try multiple property names that the proxy client might use
            for prop_name in ["server_id", "node_id", "client_id", "cid", "id", "partition_id"]:
                if prop_name in props:
                    return int(props[prop_name])
            return None
        except Exception as exc:
            logger.error(
                f"🔍 [DEBUG] ❌ EXCEPTION: Failed to get server_id from client {client_proxy.cid}: {exc}"
            )
            return None
    
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure fit by sending each server its own model parameters."""
        logger.info(f"🔍 [DEBUG] ===== StaticViewStrategy.configure_fit() STARTED =====")
        logger.info(f"🔍 [DEBUG] server_round: {server_round}")
        logger.info(f"🔍 [DEBUG] self.num_heads: {self.num_heads}")
        
        config = []
        
        # Get all available clients (proxy clients representing leaf servers)
        clients = list(client_manager.all().values())
        logger.info(f"🔍 [DEBUG] Found {len(clients)} clients from client_manager")
        
        for i, client_proxy in enumerate(clients):
            logger.info(f"🔍 [DEBUG] --- Processing client {i+1}/{len(clients)} ---")
            logger.info(f"🔍 [DEBUG] client_proxy.cid: {client_proxy.cid}")
            logger.info(f"🔍 [DEBUG] client_proxy type: {type(client_proxy)}")
            
            # Extract server ID from client properties using new Flower API
            logger.info(f"🔍 [DEBUG] Calling _get_server_id() with new Flower API...")
            server_id = self._get_server_id(client_proxy)
            logger.info(f"🔍 [DEBUG] Received server_id: {server_id}")
            logger.info(f"🔍 [DEBUG] Valid range check: 0 <= {server_id} < {self.num_heads} = {0 <= server_id < self.num_heads if server_id is not None else 'N/A'}")
            
            if server_id is not None and 0 <= server_id < self.num_heads:
                # Send this server its own model parameters
                from flwr.common import FitIns
                fit_ins = FitIns(self.head_params[server_id], {})
                config.append((client_proxy, fit_ins))
                logger.info(f"🔍 [DEBUG] ✅ SUCCESS: Configured server {server_id} (client {client_proxy.cid}) with its own parameters")
            elif server_id is not None:
                logger.warning(f"🔍 [DEBUG] ❌ INVALID: server_id {server_id}, expected 0-{self.num_heads-1} (client {client_proxy.cid})")
            else:
                logger.error(f"🔍 [DEBUG] ❌ NO SERVER_ID: No valid server_id found for client {client_proxy.cid}")
        
        logger.info(f"🔍 [DEBUG] Final config length: {len(config)}")
        if len(config) == 0:
            logger.error(f"🔍 [DEBUG] ❌ FAILURE: configure_fit: no clients selected, cancel")
        else:
            logger.info(f"🔍 [DEBUG] ✅ SUCCESS: Configured {len(config)} servers with independent parameters")
        
        logger.info(f"🔍 [DEBUG] ===== StaticViewStrategy.configure_fit() COMPLETE =====")
        return config
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results by updating each server's independent parameters."""
        logger.info(f"🔍 [DEBUG] ===== StaticViewStrategy.aggregate_fit() STARTED =====")
        logger.info(f"🔍 [DEBUG] server_round: {server_round}")
        logger.info(f"🔍 [DEBUG] results count: {len(results)}")
        logger.info(f"🔍 [DEBUG] failures count: {len(failures)}")
        
        if failures:
            logger.warning(f"🔍 [DEBUG] ⚠️ {len(failures)} servers failed during training")
        
        # Update each server's parameters independently
        for i, (client_proxy, fit_result) in enumerate(results):
            logger.info(f"🔍 [DEBUG] --- Processing result {i+1}/{len(results)} ---")
            logger.info(f"🔍 [DEBUG] client_proxy.cid: {client_proxy.cid}")
            # Extract server ID from client properties using new Flower API
            logger.info(f"🔍 [DEBUG] Calling _get_server_id() with new Flower API...")
            server_id = self._get_server_id(client_proxy)
            logger.info(f"🔍 [DEBUG] Received server_id: {server_id}")
            
            if server_id is not None and 0 <= server_id < self.num_heads:
                # Store the updated parameters for this server
                self.head_params[server_id] = fit_result.parameters
                logger.info(f"🔍 [DEBUG] Updated head_params[{server_id}] with new parameters")
                
                # Update optimizer state if available
                if hasattr(fit_result, 'metrics') and fit_result.metrics:
                    # Store any optimizer-related metrics
                    self.head_opt_state[server_id].update(fit_result.metrics)
                    logger.info(f"🔍 [DEBUG] Updated optimizer state for server {server_id}")
                
                logger.info(f"🔍 [DEBUG] ✅ SUCCESS: Updated parameters for server {server_id} (client {client_proxy.cid})")
            elif server_id is not None:
                logger.warning(f"🔍 [DEBUG] ❌ INVALID: server_id {server_id} in results (client {client_proxy.cid})")
            else:
                logger.error(f"🔍 [DEBUG] ❌ NO SERVER_ID: No valid server_id found for client {client_proxy.cid}")
        
        logger.info(f"🔍 [DEBUG] ✅ SUCCESS: Aggregated {len(results)} server results independently")
        logger.info(f"🔍 [DEBUG] ===== StaticViewStrategy.aggregate_fit() COMPLETE =====")
        
        # Save all head models to disk for persistence across global rounds
        self._save_head_models()
        
        # Flower expects a single Parameters object - return the first head's parameters
        # This is only used for logging/compatibility, actual per-server parameters are in head_params
        if self.head_params:
            return self.head_params[0], {}
        else:
            # Fallback - should not happen
            model = Net()
            return ndarrays_to_parameters(get_weights(model)), {}
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results for StaticViewStrategy."""
        if failures:
            logger.warning(f"[StaticViewStrategy] {len(failures)} servers failed during evaluation")
        
        if not results:
            return None
        
        # Calculate weighted average loss and accuracy across all servers
        # results contains (client_proxy, evaluate_res) tuples
        total_examples = sum([res.num_examples for _, res in results])
        if total_examples == 0:
            return None
            
        weighted_loss = sum([res.num_examples * res.loss for _, res in results]) / total_examples
        
        # Aggregate accuracy if available
        accuracies = [res.num_examples * res.metrics.get("accuracy", 0.0) for _, res in results]
        weighted_accuracy = sum(accuracies) / total_examples
        
        aggregated_metrics = {
            "accuracy": weighted_accuracy,
            "samples": total_examples
        }
        
        return weighted_loss, aggregated_metrics

# Global evaluation function for FedAdam
def evaluate_global_model(server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
    """Evaluate the global model on test datasets from all servers."""
    try:
        # Reconstruct global model from parameters
        model = Net()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        cluster_accuracies = {}
        
        # Get number of servers from environment or config
        num_servers = int(os.getenv('NUM_SERVERS', '3'))
        
        # Use CIFAR-10 test dataset for all servers (no view-based datasets)
        test_loader = get_cifar10_test_loader(batch_size=64)
        
        # Evaluate on CIFAR-10 test dataset for all servers
        for server_id in range(num_servers):  # Dynamic server count
            try:
                
                # Evaluate model on this server's test data
                model.eval()
                server_loss = 0.0
                server_correct = 0
                server_samples = 0
                
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.squeeze().long().to(device)
                        outputs = model(images)
                        loss = torch.nn.functional.cross_entropy(outputs, labels)
                        server_loss += loss.item() * images.size(0)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        server_correct += (predicted == labels).sum().item()
                        server_samples += images.size(0)
                
                if server_samples > 0:
                    server_accuracy = server_correct / server_samples
                    server_avg_loss = server_loss / server_samples
                    
                    total_loss += server_loss
                    total_accuracy += server_correct
                    total_samples += server_samples
                    cluster_accuracies[f"server_{server_id}_cifar10"] = server_accuracy
                    
                    logger.info(f"[Cloud Eval] Server {server_id} (CIFAR-10): {server_accuracy:.4f} acc, {server_avg_loss:.4f} loss, {server_samples} samples")
                
            except Exception as e:
                logger.warning(f"[Cloud Eval] Failed to evaluate server {server_id}: {e}")
                continue
        
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_accuracy = total_accuracy / total_samples
            
            logger.info(f"[Cloud Eval] Global: {avg_accuracy:.4f} accuracy, {avg_loss:.4f} loss, {total_samples} samples")
            logger.info(f"[Cloud Eval] Cluster accuracies: {cluster_accuracies}")
            
            # Write validation loss for early stopping
            try:
                # Get global round from environment or use server_round
                global_round = int(os.environ.get("GLOBAL_ROUND", server_round))
                
                # Create global directory structure
                global_dir = Path(f"rounds/round_{global_round}/global")
                global_dir.mkdir(parents=True, exist_ok=True)
                
                # Write val_loss.txt to correct location
                val_loss_path = global_dir / "val_loss.txt"
                with open(val_loss_path, "w") as f:
                    f.write(f"{avg_loss:.6f}\n")
                logger.info(f"[Cloud Eval] Written validation loss to {val_loss_path}")
            except Exception as e:
                logger.warning(f"[Cloud Eval] Failed to write val_loss.txt: {e}")
            
            # Return metrics in the format expected by Flower
            metrics = {
                "accuracy": avg_accuracy,
                "samples": total_samples,
                **{f"cluster_acc_{k}": v for k, v in cluster_accuracies.items()}
            }
            
            return avg_loss, metrics
        else:
            logger.warning("[Cloud Eval] No test samples found")
            return None
            
    except Exception as e:
        logger.error(f"[Cloud Eval] Evaluation failed: {e}")
        return None

class CloudFedAdam(FedAdam):
    """Enhanced FedAdam strategy with SCAFFOLD, dynamic clustering, and evaluation."""
    
    def __init__(self, **kwargs):
        # Extract and remove scaffold/dynamic flags
        scaffold_flag = kwargs.pop('scaffold_enabled', False)
        dynamic_flag = kwargs.pop('dynamic_clustering', False)
        # Extract optional model reference for advanced algorithms
        model_ref = kwargs.pop('model', None)
        # Extract clustering configuration dict if provided
        cluster_cfg = kwargs.pop('clustering_config', None)
        # Pass evaluation function to parent
        kwargs['evaluate_fn'] = evaluate_global_model
        super().__init__(**kwargs)
        
        # Initialize SCAFFOLD if enabled
        self._scaffold_manager = None
        if scaffold_flag and model_ref is not None:
            self._scaffold_manager = create_scaffold_manager(model_ref) if create_scaffold_manager else None
            if self._scaffold_manager:
                logger.info("[Cloud Server] SCAFFOLD control variates initialized")
        
        # Initialize dynamic clustering if enabled
        self._dynamic_clustering = None
        if dynamic_flag and cluster_cfg is not None:
            self._dynamic_clustering = create_dynamic_clustering(cluster_cfg) if create_dynamic_clustering else None
            if self._dynamic_clustering:
                logger.info("[Cloud Server] Dynamic clustering initialized")
    
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure fit with SCAFFOLD server control variates."""
        # Get base configuration from parent
        config = super().configure_fit(server_round, parameters, client_manager)
        
        # Add SCAFFOLD server control variates if enabled
        if self._scaffold_manager is not None:
            try:
                # Get current server control variates
                server_control = self._scaffold_manager.get_server_control()
                
                # Serialize for transmission
                serialized_control = base64.b64encode(pickle.dumps(server_control)).decode('utf-8')
                
                # Add to all client configs
                for client_proxy, fit_ins in config:
                    fit_ins.config["scaffold_server_control"] = serialized_control
                
                logger.debug(f"[Cloud Server] SCAFFOLD server control variates broadcasted to {len(config)} leaf servers")
                
            except Exception as e:
                logger.error(f"[Cloud Server] Failed to broadcast SCAFFOLD control variates: {e}")
        
        return config

def run_server():
    """Run the cloud server with enhanced FedAdam strategy."""
    try:
        # Log environment variables set by orchestrator
        global_round = os.getenv('GLOBAL_ROUND', '1')
        num_servers = os.getenv('NUM_SERVERS', '3')
        logger.info(f"[Cloud Server] Starting for global round {global_round}")
        logger.info(f"[Cloud Server] Expecting {num_servers} leaf servers")
        
        # Load configuration
        config_path = Path("pyproject.toml")
        if not config_path.exists():
            raise FileNotFoundError("pyproject.toml not found")
        
        config = toml.load(config_path)
        
        # Extract configuration
        hierarchy_config = config.get("tool", {}).get("flwr", {}).get("hierarchy", {})
        optimizer_config = config.get("tool", {}).get("flwr", {}).get("optimizer", {})
        cluster_config = config.get("tool", {}).get("flwr", {}).get("cluster", {})
        
        # Create initial parameters for FedAdam
        init_model = Net()
        initial_parameters = ndarrays_to_parameters(get_weights(init_model))

        # Get number of servers from environment (set by orchestrator)
        num_servers = int(os.getenv('NUM_SERVERS', '3'))
        
        # Get dataset flag and clustering mode from environment (set by orchestrator)
        dataset_flag = hierarchy_config.get("dataset_flag", "cifar10")
        # Override dynamic_clustering with environment variable (orchestrator enforces correct mode)
        dynamic_clustering_env = os.getenv("DYNAMIC_CLUSTERING", "true").lower() == "true"
        logger.info(f"[Cloud Server] Dataset mode: {dataset_flag}, Dynamic clustering: {dynamic_clustering_env}")
        
        # Use CloudFedAdam for CIFAR-10 with dynamic clustering
        strategy = CloudFedAdam(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_servers,
            min_evaluate_clients=num_servers,
            min_available_clients=num_servers,
            initial_parameters=initial_parameters,
            eta=0.5,
            eta_l=0.5,
            beta_1=0.95,
            beta_2=0.999,
            tau=0.001,
            model=init_model,
            clustering_config=cluster_config
        )
        logger.info(f"[Cloud Server] Using CloudFedAdam for CIFAR-10 with dynamic weight-based clustering")
        
        # Create start signal file in signals directory (absolute path)
        signal_path = Path().resolve() / "signals" / "cloud_started.signal"
        create_signal_file(signal_path, "Cloud server started")
        
        # Get server rounds per global round from configuration
        server_rounds = hierarchy_config.get('server_rounds_per_global', 1)
        
        # Start server
        logger.info(f"[Cloud Server] Starting on 0.0.0.0:{hierarchy_config.get('cloud_port', 6000)}")
        logger.info(f"[Cloud Server] Running {server_rounds} server rounds per global round")
        
        # Temporarily allow output to debug hanging issue
        import contextlib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Ensure server waits for all 3 proxy clients before starting rounds
            MIN_CLIENTS = num_servers  # Always expect all proxy clients
            
            history = start_server(
                server_address=f"0.0.0.0:{hierarchy_config.get('cloud_port', 6000)}",
                config=ServerConfig(
                    num_rounds=server_rounds,
                    min_available_clients=MIN_CLIENTS,
                    min_fit_clients=MIN_CLIENTS,
                    min_evaluate_clients=MIN_CLIENTS,
                ),
                strategy=strategy
            )
        
        # Create completion signal in signals directory (absolute path)
        completion_signal_path = Path().resolve() / "signals" / "cloud_completed.signal"
        create_signal_file(completion_signal_path, "Cloud server completed")
        
        logger.info("[Cloud Server] Training completed successfully")
        return history
        
    except Exception as e:
        logger.error(f"[Cloud Server] Failed to start: {e}")
        raise

if __name__ == "__main__":
    run_server()
