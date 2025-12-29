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
from typing import List, Tuple, Optional, Any
from pathlib import Path
import toml
from fedge.utils import fs

from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg

from fedge.task import Net, get_weights, set_weights, load_data, test, set_global_seed
import torch

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
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
import sys
sys.stdout = _DropDeprecated(sys.stdout)
sys.stderr = _DropDeprecated(sys.stderr)


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
        # Pass most args to FedAvg base class
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=clients_per_server,
            initial_parameters=initial_parameters,
            #fit_metrics_aggregation_fn=self.weighted_average,
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )
        self.server_id = server_id
        self.num_rounds = num_rounds
        self.server_str = f"Leaf Server {server_id}"
        self.global_round = global_round
        # Store config for local-central evaluation
        self.clients_per_server = clients_per_server
        try:
            cfg = toml.load((Path(__file__).resolve().parent.parent) / "pyproject.toml")
            self.num_servers = cfg["tool"]["flwr"]["hierarchy"]["num_servers"]
        except Exception:
            self.num_servers = 1
        # Flat per-server directory
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        self.base_dir = fs.leaf_server_dir(project_root, server_id)
        (self.base_dir / "models").mkdir(parents=True, exist_ok=True)

    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Weighted average using per-client test_accuracy (baseline)
        vals = []
        total_examples = 0
        for num_examples, m in metrics:
            acc = m.get("test_accuracy") if isinstance(m, dict) else None
            if acc is None:
                acc = m.get("accuracy", 0.0)
            vals.append(num_examples * acc)
            total_examples += num_examples
        return {"accuracy": (sum(vals) / max(1, total_examples))}

    def _write_fit_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        """Save per-client FIT metrics with baseline key names."""
        local_rnd = rnd - 1
        clients_csv = self.base_dir / "client_fit_metrics.csv"
        write_header = not clients_csv.exists()
        with open(clients_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "global_round","local_round","client_id",
                    "client_train_loss_mean","client_train_accuracy_mean",
                    "download_bytes","upload_bytes","comp_time_sec",
                    "num_inner_batches","total_train_samples","num_examples",
                ],
            )
            if write_header:
                writer.writeheader()
            for cid, fit_res in results:
                cid_str = fit_res.metrics.get("client_id", str(cid))
                n = fit_res.num_examples
                writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "client_id": cid_str,
                    "client_train_loss_mean": fit_res.metrics.get("train_loss_mean", None),
                    "client_train_accuracy_mean": fit_res.metrics.get("train_accuracy_mean", None),
                    "download_bytes": int(fit_res.metrics.get("download_bytes", 0)),
                    "upload_bytes": int(fit_res.metrics.get("upload_bytes", 0)),
                    "comp_time_sec": float(fit_res.metrics.get("comp_time_sec", 0.0)),
                    "num_inner_batches": int(fit_res.metrics.get("num_inner_batches", 0)),
                    "total_train_samples": int(fit_res.metrics.get("total_train_samples", n)),
                    "num_examples": n,
                })
        logger.info(f"[{self.server_str} | Round {local_rnd}] Appended client fit metrics → {clients_csv}")

    def _write_eval_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        """Save per-client evaluation loss & accuracy for this round (baseline keys)."""
        local_rnd = rnd - 1
        clients_eval_csv = self.base_dir / "client_eval_metrics.csv"
        write_header = not clients_eval_csv.exists()
        with open(clients_eval_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["global_round","local_round","client_id","client_test_loss","client_test_accuracy","num_examples"])
            if write_header:
                writer.writeheader()
            for cid, eval_res in results:
                cid_str = eval_res.metrics.get("client_id", str(cid))
                # Flower passes loss separately; accuracy is under 'test_accuracy'
                loss = getattr(eval_res, "loss", None)
                if loss is None:
                    loss = eval_res.metrics.get("test_loss")
                acc = eval_res.metrics.get("test_accuracy", eval_res.metrics.get("accuracy"))
                n = eval_res.num_examples
                writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "client_id": cid_str,
                    "client_test_loss": loss,
                    "client_test_accuracy": acc,
                    "num_examples": n,
                })
        logger.info(f"[{self.server_str} | Round {local_rnd}] Appended client eval metrics → {clients_eval_csv}")

    def aggregate_fit(self, rnd, results, failures):
        
        # Print per-client train_loss and num_examples
        for cid, fit_res in results:
            friendly = fit_res.metrics.get("client_id", str(cid))
            loss = fit_res.metrics.get("train_loss", None)
            n = fit_res.num_examples
            
        if failures:
            for failure in failures:
                pass
                
        # Dump current round client metrics
        self._write_fit_metrics_csv(rnd, results)
        start_time = time.perf_counter()
        aggregated = super().aggregate_fit(rnd, results, failures)
        end_time = time.perf_counter()
        # Record server communication and computation metrics (sum of client fit)
        self.server_bytes_down = sum(int(fit_res.metrics.get("download_bytes", 0)) for _, fit_res in results)
        self.server_bytes_up = sum(int(fit_res.metrics.get("upload_bytes", 0)) for _, fit_res in results)
        self.server_comp_time = end_time - start_time
        self.server_round_time = end_time - start_time

        if aggregated is not None and isinstance(aggregated, tuple) and len(aggregated) > 0:
            self.latest_parameters = aggregated[0]

        if rnd == self.num_rounds and aggregated is not None:
            # Calculate total number of training examples from this round's results
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            

            # Extract parameters from the returned tuple
            if isinstance(aggregated, tuple) and len(aggregated) > 0:
                parameters = aggregated[0]
            else:
                # Fallback if the return type is not a tuple
                parameters = aggregated

            # Save model in flat models/ directory
            model_dir = self.base_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"model_s{self.server_id}_g{self.global_round}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump((parameters_to_ndarrays(parameters), total_examples), f)
            

        return aggregated

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

        # 3) Server evaluation on union of this server's client test shards (from PARTITIONS_JSON)
        local_rnd = rnd - 1
        model = Net()
        if hasattr(self, "latest_parameters") and self.latest_parameters is not None:
            nds = parameters_to_ndarrays(self.latest_parameters)
            set_weights(model, nds)
        device = torch.device("cpu")
        # Load indices from mandatory PARTITIONS_JSON
        import json
        parts_path = os.environ.get("PARTITIONS_JSON")
        assert parts_path is not None and Path(parts_path).exists(), "PARTITIONS_JSON is required and must exist"
        with open(parts_path, "r", encoding="utf-8") as fp:
            combined = json.load(fp)
        assert "test" in combined, "PARTITIONS_JSON must contain 'test' mapping"
        server_map_test = combined["test"][str(self.server_id)]
        indices_test = [idx for lst in server_map_test.values() for idx in lst]
        # Build test loader as union for this server
        _, testloader_local, _ = load_data("cifar10", 0, 1, batch_size=32, indices_test=indices_test)
        l_loss, l_acc = test(model, testloader_local, device)
        n = len(testloader_local.dataset)
        server_partition_loss = float(l_loss) if n > 0 else None
        server_partition_acc = float(l_acc) if n > 0 else None

        # 4) Write server-level metrics CSV with better column names
        server_csv = self.base_dir / "server_metrics.csv"
        write_header = not server_csv.exists()
        with open(server_csv, "a", newline="") as fsv:
            fieldnames = [
                "global_round", "local_round",
                "avg_test_loss_participating_clients", "avg_test_accuracy_participating_clients",
                "server_test_loss_on_server_partitions", "server_test_accuracy_on_server_partitions",
                "bytes_down", "bytes_up", "comp_time", "round_time",
            ]
            writer = csv.DictWriter(fsv, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": self.global_round,
                "local_round": local_rnd,
                "avg_test_loss_participating_clients": agg_loss,
                "avg_test_accuracy_participating_clients": agg_acc,
                "server_test_loss_on_server_partitions": server_partition_loss,
                "server_test_accuracy_on_server_partitions": server_partition_acc,
                "bytes_down": getattr(self, "server_bytes_down", 0),
                "bytes_up": getattr(self, "server_bytes_up", 0),
                "comp_time": getattr(self, "server_comp_time", 0.0),
                "round_time": getattr(self, "server_round_time", 0.0),
            })
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
    
    # Process-level seeding for reproducibility
    try:
        cfg_seed = toml.load((Path(__file__).resolve().parent.parent) / "pyproject.toml")["tool"]["flwr"]["app"]["config"].get("seed", 0)
        set_global_seed(int(cfg_seed) + int(server_id))
    except Exception:
        pass

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
        history = start_server(
            server_address=f"{bind_addr}:{args.port}",
            config=ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
        )
        logger.info(f"[{server_str}] Server has completed all rounds successfully")
        # ------------------------------------------------------------------
        # Persist per-round communication metrics
        # ------------------------------------------------------------------
        try:
            import pandas as pd, filelock
            mdf = getattr(history, "metrics_distributed_fit", {}) or {}
            def _collect(keys):
                for k in keys:
                    if k in mdf:
                        return mdf[k]
                return []
            up_entries = _collect(["bytes_up", "bytes_written"])
            down_entries = _collect(["bytes_down", "bytes_read"])
            # Evaluation-phase download traffic (e.g. parameters pushed during evaluate())
            eval_down_entries = _collect(["bytes_down_eval"])
            rt_entries = _collect(["round_time"])
            comp_entries = _collect(["compute_s"])
            by_round: dict[int, dict[str, float]] = {}
            for rnd, val in up_entries:
                by_round.setdefault(rnd, {})["bytes_up"] = int(val)
            for rnd, val in down_entries:
                by_round.setdefault(rnd, {})["bytes_down"] = int(val)
            # Merge evaluation download into totals and keep separate field
            for rnd, val in eval_down_entries:
                by_round.setdefault(rnd, {})["bytes_down_eval"] = int(val)
                by_round[rnd]["bytes_down"] = int(val) + int(by_round[rnd].get("bytes_down", 0))
            for rnd, val in rt_entries:
                by_round.setdefault(rnd, {})["round_time"] = float(val)
            for rnd, val in comp_entries:
                by_round.setdefault(rnd, {})["compute_s"] = float(val)
            rows = [
                {"global_round": args.global_round,
                 "round": rnd,
                 "bytes_up": int(vals.get("bytes_up", 0)),
                 "bytes_down": int(vals.get("bytes_down", 0)),
                 "bytes_down_eval": int(vals.get("bytes_down_eval", 0)),
                 "round_time": vals.get("round_time", 0.0),
                 "compute_s": vals.get("compute_s", 0.0)}
                for rnd, vals in sorted(by_round.items())]
            if rows:
                df = pd.DataFrame(rows)
                out = Path(os.getenv("RUN_DIR", ".")) / f"edge_comm_{server_id}.csv"
                with filelock.FileLock(out.with_suffix(".lock")):
                    mode = "a" if out.exists() else "w"
                    df.to_csv(out, index=False, mode=mode, header=not out.exists())
                logger.info(f"[{server_str}] Wrote communication CSV → {out}")
        except Exception as csv_err:
            logger.warning(f"[{server_str}] Could not write comm CSV: {csv_err}")
    except KeyboardInterrupt:
        logger.info(f"[{server_str}] Server interrupted by user")
    except Exception as e:
        logger.error(f"[{server_str}] Server error: {e}")
        # Try to save the model in case of unexpected error
        try:
            if hasattr(strategy, "parameters") and strategy.parameters is not None:
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent
                model_dir = project_root / "models"
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / f"server_{server_id}.pkl"
                
                with open(model_path, "wb") as f:
                    pickle.dump((parameters_to_ndarrays(strategy.parameters), 0), f)
                logger.info(f"[{server_str}] Saved error recovery model to {model_path}")
        except Exception as save_error:
            logger.error(f"[{server_str}] Could not save error recovery model: {save_error}")


if __name__ == "__main__":
    main()
