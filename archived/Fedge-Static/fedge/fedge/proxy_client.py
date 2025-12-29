# proxy_client.py
import pickle
import argparse
import os
import time
import warnings
import sys
import signal
from pathlib import Path
from flwr.client import NumPyClient, start_client
# ClientConfig is only available in newer Flower versions
try:
    from flwr.client import ClientConfig  # type: ignore
except ImportError:  # pragma: no cover
    ClientConfig = None  # fallback for Flower versions without ClientConfig

# Import new Flower API classes
try:
    from flwr.common import PropertiesIns, PropertiesRes
except ImportError:
    PropertiesIns = None
    PropertiesRes = None
           
            # Suppress Flower deprecation warnings during client startup
import contextlib
import io
from flwr.common import NDArrays
from typing import Tuple, Dict, Optional
import atexit
import torch
import toml
import grpc
import logging
from fedge.task import Net, set_weights, test, get_weights, get_cifar10_test_loader
from fedge.utils import fs
from fedge.utils.bytes_helper import raw_bytes
# ClientConfig not available in this Flower version - using get_properties() method instead


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Suppress Python deprecation warnings in flwr module
import sys, logging, warnings
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

class ProxyClient(NumPyClient):
    """Proxy client that uploads leaf server models to cloud server."""
    
    def __init__(self, server_id):
        self.server_id = server_id
        self.proxy_id = os.environ.get("PROXY_ID", f"proxy_{server_id}")
        self._sent_complete = False  # prevent duplicates

        # Use absolute path to project root (consistent with orchestrator)
        project_root = Path(__file__).resolve().parent.parent

        # Current global round (1-indexed for directory structure)
        global_round = int(os.environ.get("GLOBAL_ROUND", "1"))

        # Standard directory structure: rounds/round_X/leaf/server_Y/
        self.base_dir = project_root / "rounds" / f"round_{global_round}" / "leaf" / f"server_{server_id}"
        models_dir = self.base_dir / "models"

        # Path to model for this global round; fallback to latest model if not present
        model_path = models_dir / f"model_s{server_id}_g{global_round}.pkl"
        if not model_path.exists():
            # Wait (up to 300 s) for the leaf server to save the model
            wait_sec, max_wait = 0, 300
            logger.info(f"[{self.proxy_id}] Waiting for model file {model_path} (max {max_wait}s)")
            while not model_path.exists() and wait_sec < max_wait:
                time.sleep(2)
                wait_sec += 2
            if not model_path.exists():
                logger.error(f"[{self.proxy_id}] Model file not found after {max_wait}s – aborting")
                sys.exit(1)

        # Fallback to newest model pkl in directory if specific round file still missing
        if not model_path.exists():
            pkl_files = sorted(models_dir.glob("*.pkl"))
            if pkl_files:
                model_path = pkl_files[-1]
        print(f"[{self.proxy_id}] Loading model from {model_path}")

        # Try to load the model with error handling for corrupted files
        loaded_data = None
        try:
            # Check if file is not empty
            if model_path.stat().st_size == 0:
                raise ValueError(f"Model file {model_path} is empty")
                
            with open(model_path, "rb") as f:
                # Load both NDArrays list and total samples from pickle
                loaded_data = pickle.load(f)

                # Handle both formats (for backward compatibility)
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    self.ndarrays, self.total_examples = loaded_data
                    print(f"[{self.proxy_id}] Loaded model and {self.total_examples} total training examples")
                else:
                    # Fall back to old format if needed
                    self.ndarrays = loaded_data
                    self.total_examples = 1  # Default to 1 for backward compatibility
                    print(f"[{self.proxy_id}] Loaded model (no sample count available)")
                    
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            print(f"[{self.proxy_id}] Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Cannot proceed with corrupted/missing model file {model_path}: {e}") from e

        # Initialize model and validation data for evaluation (CIFAR-10 only)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        set_weights(self.net, self.ndarrays)
        # Use CIFAR-10 dedicated test loader for global evaluation
        self.valloader = get_cifar10_test_loader(batch_size=128)

        # Ensure completion CSV row on exit
        atexit.register(self._write_completion_signal)

        # Proxy signals CSV in standard leaf server directory
        # Ensure directory exists before writing
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.proxy_signals_csv = self.base_dir / "proxy_signals.csv"
        write_header = not self.proxy_signals_csv.exists()
        with open(self.proxy_signals_csv, "a", newline="") as fcsv:
            import csv
            writer = csv.DictWriter(fcsv, fieldnames=["global_round","proxy_id","server_id","signal_type","timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": global_round,
                "proxy_id": self.proxy_id,
                "server_id": server_id,
                "signal_type": "started",
                "timestamp": time.time(),
            })
        print(f"[{self.proxy_id}] Wrote start row to {self.proxy_signals_csv}")

    def get_properties(self, ins):
        """Return client properties including deterministic client_id.
        
        Supports both old (dict) and new (PropertiesIns) Flower API versions.
        This is a backup method in case FLWR_CLIENT_NODE_ID doesn't work.
        We try multiple property names that different Flower versions might use.
        """
        # Accept both old (dict) and new (PropertiesIns) calls
        if PropertiesIns and isinstance(ins, PropertiesIns):
            cfg = ins.config
            api_version = "new (PropertiesIns)"
        else:
            cfg = ins or {}
            api_version = "old (dict)"
        
        client_id = str(self.server_id)
        logger.info(f"🔍 [DEBUG] [{self.proxy_id}] ===== get_properties() CALLED =====")
        logger.info(f"🔍 [DEBUG] [{self.proxy_id}] API version: {api_version}")
        logger.info(f"🔍 [DEBUG] [{self.proxy_id}] Input config: {cfg}")
        logger.info(f"🔍 [DEBUG] [{self.proxy_id}] self.server_id: {self.server_id}")
        logger.info(f"🔍 [DEBUG] [{self.proxy_id}] client_id (str): {client_id}")
        
        # Return multiple property formats
        properties = {
            "client_id": client_id,
            "node_id": client_id,
            "cid": client_id,
            "id": client_id,
            "partition_id": client_id,
            "server_id": client_id,
            **cfg,  # echo any extra fields
        }
        logger.info(f"🔍 [DEBUG] [{self.proxy_id}] Returning properties: {properties}")
        
        # Return appropriate format based on API version
        if PropertiesIns and isinstance(ins, PropertiesIns):
            logger.info(f"🔍 [DEBUG] [{self.proxy_id}] ===== get_properties() COMPLETE (PropertiesRes) =====")
            return PropertiesRes(properties=properties)
        else:
            logger.info(f"🔍 [DEBUG] [{self.proxy_id}] ===== get_properties() COMPLETE (dict) =====")
            return properties

    def get_parameters(self, config) -> NDArrays:
        print(f"[{self.proxy_id}] Providing leaf server parameters to cloud")
        return self.ndarrays

    def fit(self, parameters, config) -> Tuple[NDArrays, int, dict]:
        import time
        t0 = time.time()
        print(f"[{self.proxy_id}] Received fit request, re-sending edge model with {self.total_examples} samples")
        bytes_up = raw_bytes(self.ndarrays)
        bytes_down = raw_bytes(parameters)  # size of global model downloaded from cloud
        round_time = time.time() - t0  # negligible but keeps schema consistent
        # Add server metadata for stable clustering
        dataset_flag = "cifar10"
        
        return self.ndarrays, self.total_examples, {
            "sid": self.server_id,
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "round_time": round_time,
            "compute_s": 0.0,
            # Server metadata for clustering (flattened to avoid nested dict issues)
            "server_id": self.server_id,
            "dataset_flag": dataset_flag,
            "view_type": "cifar10_server",
            "specialization": "cifar10_dynamic"
        }

    def evaluate(self, parameters, config) -> Tuple[float, int, dict]:
        # Update model weights from server and run evaluation
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        num_examples = len(self.valloader.dataset)
        print(f"[{self.proxy_id}] Eval -> loss: {loss}, samples: {num_examples}, accuracy: {accuracy}")

        # Return evaluation results first
        bytes_down = raw_bytes(parameters)
        metrics = {"accuracy": accuracy, "bytes_down_eval": bytes_down}
        results = (loss, num_examples, metrics)

        # Decide whether this evaluation corresponds to the *final* server
        # round executed by the current cloud instance. Flower passes the
        # current round index in `config` (1-indexed).  The orchestrator
        # injects the env-var TOTAL_SERVER_ROUNDS_THIS_CLOUD so that the
        # proxy can compare and only exit after the very last round.

        total_rounds_env = os.environ.get("TOTAL_SERVER_ROUNDS_THIS_CLOUD")
        # Fallback for backward compatibility - try multiple env var names
        if total_rounds_env is None:
            total_rounds_env = (
                os.environ.get("SERVER_ROUNDS_PER_GLOBAL")
                or os.environ.get("SERVER_ROUNDS")
                or "1"
            )

        try:
            total_rounds = int(total_rounds_env)
        except ValueError:
            total_rounds = 1

        current_round = int(config.get("server_round", 0))
        
        # Debug: Show exit decision values
        print(f"[{self.proxy_id}] Round {current_round}/{total_rounds}, should_exit={current_round >= total_rounds}")
        logger.info(f"[{self.proxy_id}] Round {current_round}/{total_rounds}, should_exit={current_round >= total_rounds}")

        should_exit = current_round >= total_rounds

        if should_exit:
            # Write completion row and exit AFTER returning results
            import threading

            def exit_after_delay():
                # Avoid duplicate signals
                if self._sent_complete:
                    return
                time.sleep(1)  # Ensure response is sent
                self._write_completion_signal()
                import os
                print(f"[{self.proxy_id}] Exiting proxy client after evaluation...")
                os._exit(0)

            # Start exit thread so we return results first
            threading.Thread(target=exit_after_delay, daemon=True).start()

        # Return results before the thread exits
        return results

    def _write_completion_signal(self):
        import csv
        global_round = int(os.environ.get("GLOBAL_ROUND", "0"))
        write_header = not self.proxy_signals_csv.exists()
        with open(self.proxy_signals_csv, "a", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=["global_round","proxy_id","server_id","signal_type","timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": global_round,
                "proxy_id": self.proxy_id,
                "server_id": self.server_id,
                "signal_type": "complete",
                "timestamp": time.time(),
            })
        self._sent_complete = True
        print(f"[{self.proxy_id}] Wrote completion row to {self.proxy_signals_csv}")

def handle_signal(sig, frame):
    """Handle termination signals gracefully"""
    proxy_id = os.environ.get("PROXY_ID", "proxy")
    logger.info(f"[{proxy_id}] Received signal {sig}, shutting down gracefully...")
    # Append completion row on signal
    server_id = int(os.environ.get("SERVER_ID", "0"))
    global_round = int(os.environ.get("GLOBAL_ROUND", "0"))
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    proxy_signals_csv = fs.leaf_server_dir(project_root, server_id, global_round) / "proxy_signals.csv"
    import csv
    with open(proxy_signals_csv, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["global_round","proxy_id","server_id","signal_type","timestamp"])
        if fp.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "global_round": global_round,
            "proxy_id": proxy_id,
            "server_id": server_id,
            "signal_type": "complete",
            "timestamp": time.time(),
        })
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--cloud_address", default=os.getenv("CLOUD_ADDRESS", "127.0.0.1:6000"))
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum connection retry attempts")
    parser.add_argument("--retry_delay", type=int, default=2, help="Seconds to wait between retries")
    parser.add_argument("--global_round", type=int, default=0, help="Current global round number (0-indexed)")
    parser.add_argument("--dir_round", type=int, help="Round number for directory structure (1-indexed)")

    args = parser.parse_args()

    # Store server_id in environment for signal handlers
    os.environ["SERVER_ID"] = str(args.server_id)

    # Store global round and dir round if provided via command line
    if args.global_round is not None:
        os.environ["GLOBAL_ROUND"] = str(args.global_round)
    if args.dir_round is not None:
        os.environ["DIR_ROUND"] = str(args.dir_round)

    # Extract ID from environment or use default
    proxy_id = os.environ.get("PROXY_ID", f"proxy_{args.server_id}")
    os.environ["PROXY_ID"] = proxy_id

    # Use the already configured logger from the top of the file
    # (logging is already set up with basicConfig)

    # ------------------------------------------------------------------ #
    # Tell Flower what the deterministic client id should be
    #   (must be a str, NOT an int)
    # ------------------------------------------------------------------ #
    client_node_id = str(args.server_id)
    os.environ["FLWR_CLIENT_NODE_ID"] = client_node_id
    
    logger.info(f"[{proxy_id}] Starting proxy client for Leaf Server {args.server_id}")
    logger.info(f"[{proxy_id}] Will connect to cloud server at {args.cloud_address}")
    logger.info(f"[{proxy_id}] Using deterministic node_id {args.server_id} via FLWR_CLIENT_NODE_ID for StaticViewStrategy")
    logger.info(f"[{proxy_id}] ClientConfig available: {ClientConfig is not None}")
    if ClientConfig is not None:
        logger.info(f"[{proxy_id}] Using ClientConfig with node_id: {client_node_id}")
    else:
        logger.info(f"[{proxy_id}] Using get_properties() fallback for node_id: {client_node_id}")

    # Pre-load model and dataset once
    client = ProxyClient(args.server_id)
    server_address = args.cloud_address

    # Wait for cloud to be ready: look for cloud_started.signal
    # Use absolute path to project root for consistency
    project_root = Path(__file__).resolve().parent.parent
    start_signal = project_root / "signals" / "cloud_started.signal"
    wait_secs = 0
    while not start_signal.exists() and wait_secs < args.retry_delay * args.max_retries:
        logger.info(f"[{proxy_id}] Waiting for cloud start signal...")
        time.sleep(1)
        wait_secs += 1

    # Run the client with retry logic (only RPC loop)
    for attempt in range(1, args.max_retries + 1):
        try:
            # Ensure FLWR_CLIENT_NODE_ID is set fresh for each attempt
            # Try multiple environment variable names that different Flower versions might use
            os.environ["FLWR_CLIENT_NODE_ID"] = client_node_id
            os.environ["FLWR_NODE_ID"] = client_node_id  # Alternative name
            os.environ["CLIENT_NODE_ID"] = client_node_id  # Alternative name
            logger.info(f"[{proxy_id}] Attempt {attempt}: Setting multiple node_id env vars to {client_node_id}")
            
            
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                # Environment variables should be read by start_client
                # Use ClientConfig if available, otherwise rely on env var + get_properties()
                start_client(
                    server_address=server_address,
                    client=client,
                    **({"client_config": ClientConfig(node_id=client_node_id)} if ClientConfig is not None else {})
                )
            logger.info(f"[{proxy_id}] Finished as Flower client {args.server_id}")
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                if attempt < args.max_retries:
                    logger.warning(f"[{proxy_id}] Connection failed (attempt {attempt}/{args.max_retries}): {e.details()}")
                    logger.info(f"[{proxy_id}] Retrying in {args.retry_delay} seconds...")
                    time.sleep(args.retry_delay)
                    continue
                else:
                    logger.error(f"[{proxy_id}] Connection failed after {args.max_retries} attempts: {e.details()}")
                    client._write_completion_signal()
                    sys.exit(1)
            else:
                logger.error(f"[{proxy_id}] Unexpected gRPC error: {e.details()}")
                client._write_completion_signal()
                sys.exit(1)
        except Exception as e:
            logger.error(f"[{proxy_id}] Unexpected error: {e}")
            client._write_completion_signal()
            sys.exit(1)
    else:
        # This else runs if the loop didn't break
        logger.error(f"[{proxy_id}] Failed to connect after {args.max_retries} attempts")
        client._write_completion_signal()
        sys.exit(1)
