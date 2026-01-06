# cloud_flower.py

import os
import time
import pickle
import toml
import threading
import sys
import signal
import warnings
import logging
import torch
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, NDArrays, Parameters, FitRes, parameters_to_ndarrays

from fedge.utils import fs
from fedge.task import ResNet18, load_data, test, set_weights, set_global_seed
import csv

# ──────────────────────────────────────────────────────────────────────────────
#  Read **all** of our hierarchy config from pyproject.toml
# ──────────────────────────────────────────────────────────────────────────────

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
cfg = toml.load(project_root / "pyproject.toml")
hier = cfg["tool"]["flwr"]["hierarchy"]

NUM_SERVERS = hier["num_servers"]
GLOBAL_ROUNDS = hier["global_rounds"]
SERVER_ROUNDS_PER_GLOBAL = hier["server_rounds_per_global"]
CLOUD_PORT = hier["cloud_port"]
CLIENTS_PER_SERVER = hier["clients_per_server"]
app_cfg = cfg["tool"]["flwr"]["app"]["config"]
SEED = int(app_cfg.get("seed", 0))
BATCH_SIZE = int(app_cfg.get("batch_size", 32))

# Require new directory structure (no fallback)
use_new_dir_structure = True

# Get the current global round from environment (0-indexed in code)
current_global_round = int(os.environ.get("GLOBAL_ROUND", "0"))

# Use 0-indexed naming consistently for directory structure
dir_round = current_global_round

# Create necessary directories
if use_new_dir_structure:
    # Create signals directory and global round directory
    signals_dir = fs.get_signals_dir(project_root)
    signals_dir.mkdir(exist_ok=True, parents=True)
    
    global_round_dir = fs.get_global_round_dir(project_root, dir_round)
    global_round_dir.mkdir(exist_ok=True, parents=True)
    
    # Paths for signals & output using new structure
    def get_global_round_signal_path(round_num: int) -> Path:
        # Note: round_num is 0-indexed for directory paths
        return fs.get_global_round_dir(project_root, round_num) / "complete.signal"

    def get_global_model_path(round_num: int) -> Path:
        # Note: round_num is 0-indexed for directory paths
        return fs.get_global_round_dir(project_root, round_num) / "model.pkl"

    # Cloud signals are in the signals directory
    start_signal = signals_dir / "cloud_started.signal"
    complete_signal = signals_dir / "cloud_complete.signal"
    
    print(f"[Cloud Server] Using consistent 0-indexed round directories")

# ──────────────────────────────────────────────────────────────────────────────
#  Helper to create a “signal file” with a timestamp
# ──────────────────────────────────────────────────────────────────────────────

def create_signal_file(file_path: Path, message: str) -> bool:
    try:
        with open(file_path, "w") as f:
            f.write(str(time.time()))
        logging.info(f"[Cloud Server] {message}: {file_path}")
        return True
    except Exception as e:
        logging.error(f"[Cloud Server] Could not create {file_path}: {e}")
        return False

# ──────────────────────────────────────────────────────────────────────────────
#  Logging / Warnings suppression
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc"):
    logging.getLogger(name).setLevel(logging.ERROR)

# Drop “DEPRECATED FEATURE” on stdout/stderr
class _DropDeprecated:
    def __init__(self, out): self._out = out
    def write(self, txt):
        if "DEPRECATED FEATURE" not in txt:
            self._out.write(txt)
    def flush(self): self._out.flush()

sys.stdout = _DropDeprecated(sys.stdout)
sys.stderr = _DropDeprecated(sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
#  Signal‐handler: If someone hits Ctrl+C, write a final cloud_complete.signal
# ──────────────────────────────────────────────────────────────────────────────

def handle_signal(sig, frame):
    cloud_id = os.environ.get("SERVER_ID", "cloud")
    logger.info(f"[{cloud_id}] Received signal {sig}, saving final completion signal...")
    create_signal_file(complete_signal, "Created completion signal from SIGTERM handler")

    # We might save a last EMERGENCY model (if desired), but skip for brevity
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ──────────────────────────────────────────────────────────────────────────────
#  Create “cloud_started.signal” right away
# ──────────────────────────────────────────────────────────────────────────────

create_signal_file(start_signal, f"Created start signal for round {current_global_round}")

# Seed this process for reproducibility
set_global_seed(SEED)

# Metrics output directory (per-seed)
metrics_dir = (project_root / "metrics" / f"seed_{SEED}")
metrics_dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Convergence tracker (same logic as FedProx baseline)
# ──────────────────────────────────────────────────────────────────────────────

class ConvergenceTracker:
    def __init__(self):
        self.prev_loss = None
        self.prev_acc = None
        self.loss_changes = []
        self.acc_changes  = []
    def update(self, round_num, loss, acc) -> dict:
        if np.isnan(loss) or np.isinf(loss):
            print(f"CONVERGENCE ERROR: loss={loss} at round {round_num}")
        if np.isnan(acc) or np.isinf(acc):
            print(f"CONVERGENCE ERROR: acc={acc} at round {round_num}")
        if self.prev_loss is None or round_num == 0:
            self.prev_loss, self.prev_acc = loss, acc
            return {
                "conv_loss_rate": 0.0,
                "conv_acc_rate": 0.0,
                "conv_loss_stability": 0.0,
                "conv_acc_stability": 0.0,
            }
        dl = loss - self.prev_loss
        da = acc  - self.prev_acc
        if np.isnan(dl) or np.isinf(dl):
            print(f"CONVERGENCE ERROR: dl={dl} (loss={loss}, prev_loss={self.prev_loss})")
        if np.isnan(da) or np.isinf(da):
            print(f"CONVERGENCE ERROR: da={da} (acc={acc}, prev_acc={self.prev_acc})")
        self.loss_changes.append(dl)
        self.acc_changes.append(da)
        self.prev_loss, self.prev_acc = loss, acc
        loss_var = float(np.var(self.loss_changes)) if len(self.loss_changes) > 1 else 0.0
        acc_var = float(np.var(self.acc_changes)) if len(self.acc_changes) > 1 else 0.0
        if np.isnan(loss_var) or np.isinf(loss_var):
            print(f"CONVERGENCE ERROR: loss_var={loss_var}, loss_changes={self.loss_changes[-5:]}")
        if np.isnan(acc_var) or np.isinf(acc_var):
            print(f"CONVERGENCE ERROR: acc_var={acc_var}, acc_changes={self.acc_changes[-5:]}")
        return {
            "conv_loss_rate":      float(dl),
            "conv_acc_rate":       float(da),
            "conv_loss_stability": loss_var,
            "conv_acc_stability":  acc_var,
        }

# ──────────────────────────────────────────────────────────────────────────────
#  Weighted‐average for leaf accuracies
# ──────────────────────────────────────────────────────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n * m.get("accuracy", 0) for n, m in metrics)
    count = sum(n for n, _ in metrics)
    if count == 0:
        return {"accuracy": 0.0}
    return {"accuracy": total / count}

# ──────────────────────────────────────────────────────────────────────────────
#  Subclass FedAvg to implement our “global‐round” logic
# ──────────────────────────────────────────────────────────────────────────────

class CloudFedAvg(FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=NUM_SERVERS,
            min_evaluate_clients=NUM_SERVERS,
            min_available_clients=NUM_SERVERS,
            #fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        self.cloud_id = os.environ.get("SERVER_ID", "cloud")
        self.current_global_round = int(os.environ.get("GLOBAL_ROUND", "0"))
        self.round_start_time = None
        self.communication_metrics = []
        self.latest_parameters: Optional[Parameters] = None
        self.conv_tracker = ConvergenceTracker()
        self.network_speed_mbps = 10.0
        logger.info(f"[Cloud Server] Initialized for GLOBAL ROUND {self.current_global_round}")

    def aggregate_fit(self, rnd: int, results: List[Tuple[str, FitRes]], failures: List[Any]):
        """Aggregate fit results from leaf servers."""
        if self.round_start_time is None:
            self.round_start_time = time.time()
        
        # Track communication metrics with ndarray sizes
        total_bytes_up = 0
        for _, res in results:
            try:
                nds = parameters_to_ndarrays(res.parameters)
                total_bytes_up += sum(arr.nbytes for arr in nds)
            except Exception:
                pass
        
        # Handle failures
        if failures:
            logger.warning(f"[Cloud] {len(failures)} proxy client(s) failed during fit")

        # Call parent aggregation
        agg = super().aggregate_fit(rnd, results, failures)
        
        # Prepare bytes_down after aggregation
        total_bytes_down = 0
        round_time = time.time() - self.round_start_time if self.round_start_time else 0.0
        if agg is None:
            return None

        # Keep latest aggregated parameters for centralized evaluation
        try:
            self.latest_parameters = agg[0]
        except Exception:
            self.latest_parameters = None

        # Each cloud_flower instance is launched fresh for exactly
        # `SERVER_ROUNDS_PER_GLOBAL` server rounds belonging to the global round
        # passed in via the `GLOBAL_ROUND` environment variable.  Therefore when
        # we finish those rounds, we should mark *that* global round complete.
        if rnd % SERVER_ROUNDS_PER_GLOBAL == 0:
            # Use the round number provided by orchestrator instead of
            # recomputing from `rnd`, which always starts at 1 for each process.
            this_global = self.current_global_round

            # 1) Save the new global model
            parameters = agg[0]
            model_path = get_global_model_path(this_global)
            with open(model_path, "wb") as f:
                pickle.dump(parameters_to_ndarrays(parameters), f)

            # Compute total bytes_down for this round (broadcast to NUM_SERVERS proxies)
            try:
                out_nds = parameters_to_ndarrays(parameters)
                total_bytes_down = sum(arr.nbytes for arr in out_nds) * NUM_SERVERS
            except Exception:
                total_bytes_down = 0

            # Append communication metrics
            self.communication_metrics.append({
                "global_round": this_global,
                "round": rnd,
                "bytes_up": int(total_bytes_up),
                "bytes_down": int(total_bytes_down),
                "round_time": float(round_time),
                "compute_s": 0.0,
            })

            # 2) Write the global_round_{this_global}_complete.signal
            round_sig = get_global_round_signal_path(this_global)
            create_signal_file(round_sig, f"Created completion signal for GLOBAL ROUND {this_global}")
            
            # 3) Write baseline CSVs mirroring FedProx
            try:
                self._write_baseline_metrics(this_global, parameters)
            except Exception as e:
                logger.error(f"[Cloud Server] Failed to write baseline metrics: {e}")
            # 3a) Persist cloud communication metrics for this global round
            try:
                self._write_cloud_comm_csv()
            except Exception as e:
                logger.warning(f"[Cloud Server] Could not write cloud_comm.csv: {e}")

            # 4) If that was the _last_ global round, write the final "cloud_complete.signal" and exit
            if this_global == GLOBAL_ROUNDS - 1:
                create_signal_file(complete_signal, "Created final completion signal for whole job")
                
                def delayed_exit():
                    time.sleep(1)
                    sys.exit(0)

                threading.Thread(target=delayed_exit, daemon=True).start()

        return agg

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Don’t launch a round of fit until we have exactly NUM_SERVERS connected.
        """
        while len(client_manager.clients) < NUM_SERVERS:
            time.sleep(0.1)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(self, rnd: int, results: List[Tuple[str, Any]], failures: List[Any]):
        """
        Just “print-and-forward” the evaluation metrics.  If we are on the *last* server-round *of the final global round*, create a final completion signal.
        """
        current_global = self.current_global_round

        if failures:
            logger.warning(f"[Cloud] {len(failures)} proxy client(s) failed during evaluate")

        # Call super and unpack exactly two values
        merged = super().aggregate_evaluate(rnd, results, failures)
        if merged is None:
            return None
        loss, metrics = merged

        # Ensure 'accuracy' exists
        if "accuracy" not in metrics:
            metrics["accuracy"] = weighted_average(
                [(r.num_examples, r.metrics) for _, r in results]
            )["accuracy"]
        metrics["num_leaf_servers"] = len(results)
        print(f"Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        # Centralized CSV writing is handled in aggregate_fit at round boundary

        # Write cloud communication CSV
        self._write_cloud_comm_csv()
        
        # Final completion signals only on the very last leaf-round of the overall training
        is_last_global_round = current_global == GLOBAL_ROUNDS - 1
        is_last_server_round = rnd == SERVER_ROUNDS_PER_GLOBAL
        if is_last_global_round and is_last_server_round:
            create_signal_file(complete_signal, "Created completion signal after final aggregate_evaluate")

        return loss, metrics

    def _write_centralized_csv(self, round_num: int, parameters: Parameters):
        # Centralized train/test eval over full dataset (partition_id=0, num_partitions=1)
        dataset_flag = "cifar10"
        trainloader, testloader, num_classes = load_data(dataset_flag, 0, 1, batch_size=BATCH_SIZE, seed=SEED)
        model = ResNet18()
        nds = parameters_to_ndarrays(parameters) if not isinstance(parameters, list) else parameters
        set_weights(model, nds)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_loss, train_acc = test(model, trainloader, device)
        test_loss, test_acc = test(model, testloader, device)
        rec = {
            "central_train_loss": float(train_loss),
            "central_train_accuracy": float(train_acc),
            "central_test_loss": float(test_loss),
            "central_test_accuracy": float(test_acc),
            "central_loss_gap": float(test_loss - train_loss),
            "central_accuracy_gap": float(train_acc - test_acc),
        }
        conv = self.conv_tracker.update(round_num, float(test_loss), float(test_acc))
        rec.update(conv)
        # Write CSV
        path = metrics_dir / "centralized_metrics.csv"
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "round",
                "central_train_loss","central_train_accuracy",
                "central_test_loss","central_test_accuracy",
                "central_loss_gap","central_accuracy_gap",
                "conv_loss_rate","conv_acc_rate","conv_loss_stability","conv_acc_stability",
            ])
            if write_header:
                writer.writeheader()
            writer.writerow({"round": round_num, **rec})

    def _write_distributed_and_fit_csvs(self, round_num: int):
        # Aggregate across all servers' per-client CSVs
        all_acc = []
        all_loss = []
        all_weights = []
        # Stable per-client maps by global client index
        per_client_acc_map: Dict[int, float] = {}
        per_client_loss_map: Dict[int, float] = {}
        # Efficiency accumulators from FIT metrics
        comp_times = []
        up_bytes = []
        down_bytes = []
        wall_times = []
        comm_times = []
        inner_batches = []
        total_train_samples = []

        for sid in range(NUM_SERVERS):
            base_dir = fs.leaf_server_dir(project_root, sid)
            # EVAL metrics
            eval_csv = base_dir / "client_eval_metrics.csv"
            with open(eval_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if int(r.get("global_round", -1)) == round_num]
            # Compute global index offset for this server (supports non-uniform CPS)
            if isinstance(CLIENTS_PER_SERVER, list):
                cps = CLIENTS_PER_SERVER[sid]
                offset = sum(CLIENTS_PER_SERVER[:sid])
            else:
                cps = CLIENTS_PER_SERVER
                offset = sid * cps
            for r in rows:
                loss = float(r["client_test_loss"]) if r["client_test_loss"] != "" else 0.0
                acc = float(r["client_test_accuracy"]) if r["client_test_accuracy"] != "" else 0.0
                n = int(r.get("num_examples", 0))
                all_acc.append(acc)
                all_loss.append(loss)
                all_weights.append(n)
                # Parse local client id from string like "leaf_{sid}_client_{cid}" (allow optional suffixes like _gr{g})
                try:
                    id_str = str(r.get("client_id", ""))
                    m = re.search(r"client_(\d+)", id_str)
                    cid_local = int(m.group(1)) if m else None
                except Exception:
                    # Fallback: skip mapping if unparsable
                    cid_local = None
                if cid_local is not None:
                    gidx = offset + cid_local
                    per_client_acc_map[gidx] = acc
                    per_client_loss_map[gidx] = loss

            # FIT metrics
            fit_csv = base_dir / "client_fit_metrics.csv"
            with open(fit_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if int(r.get("global_round", -1)) == round_num]
            for r in rows:
                comp = float(r.get("comp_time_sec", 0.0))
                dn = float(r.get("download_bytes", 0))
                up = float(r.get("upload_bytes", 0))
                comm = (dn + up) / (self.network_speed_mbps * 1e6 / 8.0) if self.network_speed_mbps > 0 else 0.0
                wall = comp + comm
                comp_times.append(comp)
                down_bytes.append(dn)
                up_bytes.append(up)
                comm_times.append(comm)
                wall_times.append(wall)
                inner_batches.append(float(r.get("num_inner_batches", 0)))
                total_train_samples.append(float(r.get("total_train_samples", 0)))

        # Weighted averages
        n_total = sum(all_weights) if all_weights else 0
        avg_acc = float(sum(w * a for w, a in zip(all_weights, all_acc)) / max(1, n_total)) if all_weights else 0.0
        avg_loss = float(sum(w * l for w, l in zip(all_weights, all_loss)) / max(1, n_total)) if all_weights else 0.0
        acc_sd = float(np.std(all_acc)) if all_acc else 0.0
        loss_sd = float(np.std(all_loss)) if all_loss else 0.0
        # Determine total number of clients for stable column ordering
        total_clients = sum(CLIENTS_PER_SERVER) if isinstance(CLIENTS_PER_SERVER, list) else (NUM_SERVERS * CLIENTS_PER_SERVER)
        n_clients = total_clients
        acc_se = acc_sd / (n_clients ** 0.5) if n_clients > 0 else 0.0
        loss_se = loss_sd / (n_clients ** 0.5) if n_clients > 0 else 0.0

        distributed = {
            "avg_accuracy": avg_acc,
            "avg_loss": avg_loss,
            "accuracy_std": acc_sd,
            "loss_std": loss_sd,
            "acc_ci95_lo": float(avg_acc - 1.96 * acc_se),
            "acc_ci95_hi": float(avg_acc + 1.96 * acc_se),
            "loss_ci95_lo": float(avg_loss - 1.96 * loss_se),
            "loss_ci95_hi": float(avg_loss + 1.96 * loss_se),
        }
        # Per-client columns (0-based contiguous, stable by global client index)
        for idx in range(total_clients):
            distributed[f"client_{idx}_accuracy"] = float(per_client_acc_map.get(idx, 0.0))
        for idx in range(total_clients):
            distributed[f"client_{idx}_loss"] = float(per_client_loss_map.get(idx, 0.0))
        # Efficiency bundle
        distributed.update({
            "avg_comp_time_sec": float(np.mean(comp_times)) if comp_times else 0.0,
            "total_comp_time_sec": float(np.sum(comp_times)) if comp_times else 0.0,
            "std_comp_time_sec": float(np.std(comp_times)) if comp_times else 0.0,
            "avg_upload_MB": (float(np.mean(up_bytes)) / 1e6) if up_bytes else 0.0,
            "total_upload_MB": (float(np.sum(up_bytes)) / 1e6) if up_bytes else 0.0,
            "avg_download_MB": (float(np.mean(down_bytes)) / 1e6) if down_bytes else 0.0,
            "total_download_MB": (float(np.sum(down_bytes)) / 1e6) if down_bytes else 0.0,
            "total_communication_MB": (float(np.sum(up_bytes) + np.sum(down_bytes)) / 1e6) if (up_bytes and down_bytes) else 0.0,
            "avg_wall_clock_sec": float(np.mean(wall_times)) if wall_times else 0.0,
            "total_wall_clock_sec": float(np.sum(wall_times)) if wall_times else 0.0,
            "std_wall_clock_sec": float(np.std(wall_times)) if wall_times else 0.0,
            "avg_comm_time_sec": float(np.mean(comm_times)) if comm_times else 0.0,
            "total_comm_time_sec": float(np.sum(comm_times)) if comm_times else 0.0,
        })
        # Write distributed CSV
        dist_path = metrics_dir / "distributed_metrics.csv"
        with open(dist_path, "a", newline="") as f:
            fieldnames = ["round"] + list(distributed.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if dist_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow({"round": round_num, **distributed})

        # FIT aggregation: compute from per-client fit values we already collected
        # We need lists of train_loss_mean and train_accuracy_mean across all clients
        train_losses = []
        train_accs = []
        for sid in range(NUM_SERVERS):
            base_dir = fs.leaf_server_dir(project_root, sid)
            fit_csv = base_dir / "client_fit_metrics.csv"
            with open(fit_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if int(r.get("global_round", -1)) == round_num]
            for r in rows:
                if r.get("client_train_loss_mean", "") != "":
                    train_losses.append(float(r["client_train_loss_mean"]))
                if r.get("client_train_accuracy_mean", "") != "":
                    train_accs.append(float(r["client_train_accuracy_mean"]))
        fit_row = {
            "avg_train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "std_train_loss": float(np.std(train_losses)) if train_losses else 0.0,
            "min_train_loss": float(np.min(train_losses)) if train_losses else 0.0,
            "max_train_loss": float(np.max(train_losses)) if train_losses else 0.0,
            "avg_train_accuracy": float(np.mean(train_accs)) if train_accs else 0.0,
            "std_train_accuracy": float(np.std(train_accs)) if train_accs else 0.0,
            "min_train_accuracy": float(np.min(train_accs)) if train_accs else 0.0,
            "max_train_accuracy": float(np.max(train_accs)) if train_accs else 0.0,
            "avg_comp_time_sec": float(np.mean(comp_times)) if comp_times else 0.0,
            "std_comp_time_sec": float(np.std(comp_times)) if comp_times else 0.0,
            "total_comp_time_sec": float(np.sum(comp_times)) if comp_times else 0.0,
            "avg_upload_MB": float(np.mean(up_bytes)) / 1e6 if up_bytes else 0.0,
            "total_upload_MB": float(np.sum(up_bytes)) / 1e6 if up_bytes else 0.0,
            "avg_download_MB": float(np.mean(down_bytes)) / 1e6 if down_bytes else 0.0,
            "total_download_MB": float(np.sum(down_bytes)) / 1e6 if down_bytes else 0.0,
            "avg_inner_batches": float(np.mean(inner_batches)) if inner_batches else 0.0,
            "total_train_samples": float(np.sum(total_train_samples)) if total_train_samples else 0.0,
        }
        fit_path = metrics_dir / "fit_metrics.csv"
        with open(fit_path, "a", newline="") as f:
            fieldnames = ["round"] + list(fit_row.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fit_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow({"round": round_num, **fit_row})

        # Optional servers_metrics.csv with per-server aggregates
        servers_row: Dict[str, Any] = {}
        for sid in range(NUM_SERVERS):
            base_dir = fs.leaf_server_dir(project_root, sid)
            cps = CLIENTS_PER_SERVER[sid] if isinstance(CLIENTS_PER_SERVER, list) else CLIENTS_PER_SERVER
            eval_csv = base_dir / "client_eval_metrics.csv"
            with open(eval_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if int(r.get("global_round", -1)) == round_num]
            accs = [float(r.get("client_test_accuracy", 0.0)) for r in rows]
            losses = [float(r.get("client_test_loss", 0.0)) for r in rows]
            fit_csv = base_dir / "client_fit_metrics.csv"
            with open(fit_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fit_rows = [r for r in reader if int(r.get("global_round", -1)) == round_num]
            comp = [float(r.get("comp_time_sec", 0.0)) for r in fit_rows]
            upb = [float(r.get("upload_bytes", 0)) for r in fit_rows]
            dnb = [float(r.get("download_bytes", 0)) for r in fit_rows]
            comms = [(u + d) / (self.network_speed_mbps * 1e6 / 8.0) if self.network_speed_mbps > 0 else 0.0 for u, d in zip(upb, dnb)]
            walls = [c + cm for c, cm in zip(comp, comms)]
            servers_row[f"server{sid}_avg_accuracy"] = float(np.mean(accs)) if accs else 0.0
            servers_row[f"server{sid}_avg_loss"] = float(np.mean(losses)) if losses else 0.0
            servers_row[f"server{sid}_client_count"] = int(len(accs))
            servers_row[f"server{sid}_avg_comp_time_sec"] = float(np.mean(comp)) if comp else 0.0
            servers_row[f"server{sid}_avg_upload_MB"] = float(np.mean(upb)) / 1e6 if upb else 0.0
            servers_row[f"server{sid}_avg_download_MB"] = float(np.mean(dnb)) / 1e6 if dnb else 0.0
            servers_row[f"server{sid}_avg_wall_clock_sec"] = float(np.mean(walls)) if walls else 0.0
            servers_row[f"server{sid}_avg_comm_time_sec"] = float(np.mean(comms)) if comms else 0.0
        if servers_row:
            srv_path = metrics_dir / "servers_metrics.csv"
            with open(srv_path, "a", newline="") as f:
                fieldnames = ["round"] + list(servers_row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if srv_path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow({"round": round_num, **servers_row})

    def _write_baseline_metrics(self, round_num: int, parameters: Parameters):
        self._write_centralized_csv(round_num, parameters)
        self._write_distributed_and_fit_csvs(round_num)
    
    def _write_cloud_comm_csv(self):
        """Write cloud communication metrics to CSV."""
        if not self.communication_metrics:
            return
        
        try:
            import pandas as pd
            df = pd.DataFrame(self.communication_metrics)
            out_path = Path(os.getenv("RUN_DIR", ".")) / "cloud_comm.csv"
            
            mode = "a" if out_path.exists() else "w"
            df.to_csv(out_path, index=False, mode=mode, header=not out_path.exists())
            logger.info(f"[Cloud Server] Wrote communication CSV → {out_path}")
        except Exception as e:
            logger.warning(f"[Cloud Server] Could not write comm CSV: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN: spin up the FedAvg server for exactly GLOBAL_ROUNDS * SERVER_ROUNDS_PER_GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_rounds = SERVER_ROUNDS_PER_GLOBAL
    config = ServerConfig(num_rounds=total_rounds)

    
    strategy = CloudFedAvg()

    try:
        bind_addr = os.getenv("BIND_ADDRESS", "0.0.0.0")
        history = start_server(
            server_address=f"{bind_addr}:{CLOUD_PORT}",
            config=config,
            strategy=strategy,
        )
        # In principle we should never get here unless the server shuts down cleanly
        if not complete_signal.exists():
            create_signal_file(complete_signal, "Created completion signal after shutdown")

        # Communication metrics are now written by the strategy itself
        pass
    except Exception as e:
        
        # Always ensure a final “complete” file
        create_signal_file(complete_signal, "Created completion signal after error")
        sys.exit(1)
