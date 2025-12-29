"""fedge: A Flower / PyTorch app (server)."""

from typing import List, Tuple, Dict, Any
import os
import csv
import numpy as np
import torch

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fedge.task import Net, get_weights, set_weights, test, set_global_seed, CIFAR_TFM  # load_data not needed here now
from .pfedme import pFedMe


# ────────────────────────── Paths / setup ──────────────────────────
os.makedirs("metrics", exist_ok=True)
METRICS_DIR = "metrics"

# Match the schemas you asked for
CENTRAL_CSV = os.path.join(METRICS_DIR, "centralized_metrics.csv")
DIST_CSV    = os.path.join(METRICS_DIR, "distributed_metrics.csv")
FIT_CSV     = os.path.join(METRICS_DIR, "fit_metrics.csv")

CENTRAL_HEADER = [
    "round",
    "central_train_loss","central_train_accuracy",
    "central_test_loss","central_test_accuracy",
    "central_loss_gap","central_accuracy_gap",
    "conv_loss_rate","conv_acc_rate",
    "conv_loss_stability","conv_acc_stability",
]

# Global strategy handle (initialized in server_fn)
strategy: pFedMe

# Round counters for logging
dist_round_counter = {"value": 1}
fit_round_counter  = {"value": 1}

# Per-round, per-client fit metrics cache: {round: {cid: metrics_dict}}
CLIENT_CACHE: Dict[int, Dict[int, Dict[str, Any]]] = {}


# ──────────────────────────── CSV helpers ──────────────────────────
def _ensure_header(path: str, header: list[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


# ───────────────────── Convergence diagnostics ─────────────────────
class ConvergenceTracker:
    def __init__(self):
        self.prev_loss = None
        self.prev_acc  = None
        self.loss_changes: List[float] = []
        self.acc_changes:  List[float] = []

    def update(self, round_num: int, loss: float, acc: float) -> dict:
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

        dl = float(loss - self.prev_loss)
        da = float(acc  - self.prev_acc)

        if np.isnan(dl) or np.isinf(dl):
            print(f"CONVERGENCE ERROR: dl={dl} (loss={loss}, prev_loss={self.prev_loss})")
        if np.isnan(da) or np.isinf(da):
            print(f"CONVERGENCE ERROR: da={da} (acc={acc}, prev_acc={self.prev_acc})")

        self.loss_changes.append(dl)
        self.acc_changes.append(da)
        self.prev_loss, self.prev_acc = loss, acc

        # Stability as variance of recent deltas (cheap and robust)
        loss_var = float(np.var(self.loss_changes)) if len(self.loss_changes) > 1 else 0.0
        acc_var  = float(np.var(self.acc_changes))  if len(self.acc_changes)  > 1 else 0.0

        if np.isnan(loss_var) or np.isinf(loss_var):
            print(f"CONVERGENCE ERROR: loss_var={loss_var}, loss_changes_tail={self.loss_changes[-5:]}")
        if np.isnan(acc_var) or np.isinf(acc_var):
            print(f"CONVERGENCE ERROR: acc_var={acc_var}, acc_changes_tail={self.acc_changes[-5:]}")

        return {
            "conv_loss_rate":      dl,
            "conv_acc_rate":       da,
            "conv_loss_stability": loss_var,
            "conv_acc_stability":  acc_var,
        }


ctracker = ConvergenceTracker()


# ───────────── Central eval: cached loaders to avoid reloads ─────────────
_CENTRAL_CACHE: Dict[str, Any] = {
    "trainloader": None,
    "testloader": None,
    "shape": None,         # (C, H, W)
    "num_classes": 10,     # CIFAR-10
    "batch_size": 32,
}


def _ensure_central_cache(central_batch_size: int = 32) -> None:
    """Initialize CIFAR-10 loaders once and cache (memory-efficient)."""
    if _CENTRAL_CACHE["trainloader"] is not None:
        return  # already initialized

    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader

    _CENTRAL_CACHE["batch_size"] = int(central_batch_size)

    train_full = CIFAR10(root="./data", train=True,  download=True, transform=CIFAR_TFM)
    test_full  = CIFAR10(root="./data", train=False, download=True, transform=CIFAR_TFM)

    trainloader = DataLoader(
        train_full,
        batch_size=_CENTRAL_CACHE["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )
    testloader  = DataLoader(
        test_full,
        batch_size=_CENTRAL_CACHE["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Infer (C,H,W) once
    sample, _ = next(iter(trainloader))
    _, in_ch, H, W = sample.shape

    _CENTRAL_CACHE["trainloader"] = trainloader
    _CENTRAL_CACHE["testloader"]  = testloader
    _CENTRAL_CACHE["shape"]       = (in_ch, H, W)


def evaluate_and_log_central(round_num: int, parameters, config):
    """Runs centralized eval on CIFAR-10 and writes centralized_metrics.csv."""
    # Ensure CIFAR-10 central loaders exist (once)
    _ensure_central_cache(config.get("central_batch_size", _CENTRAL_CACHE["batch_size"]))

    trainloader = _CENTRAL_CACHE["trainloader"]
    testloader  = _CENTRAL_CACHE["testloader"]
    in_ch, H, W = _CENTRAL_CACHE["shape"]
    num_classes = _CENTRAL_CACHE["num_classes"]

    # Build model & load global params
    net = Net(in_ch=in_ch, img_h=H, img_w=W, n_class=num_classes)

    # Accept either list of ndarrays or Flower Parameters
    try:
        nds = parameters if isinstance(parameters, list) else None
        if nds is None:
            from flwr.common import parameters_to_ndarrays as _p2n
            nds = _p2n(parameters)
    except Exception:
        from flwr.common import parameters_to_ndarrays as _p2n
        nds = _p2n(parameters)

    set_weights(net, nds)

    # Send to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Centralized train/test metrics
    train_loss, train_acc = test(net, trainloader, device)
    test_loss, test_acc   = test(net, testloader, device)

    print(
        f"[CENTRAL] Round {round_num}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
    )

    rec = {
        "central_train_loss": float(train_loss),
        "central_train_accuracy": float(train_acc),
        "central_test_loss": float(test_loss),
        "central_test_accuracy": float(test_acc),
        "central_loss_gap": float(test_loss - train_loss),
        "central_accuracy_gap": float(train_acc - test_acc),
    }

    # Convergence diagnostics
    rec.update(ctracker.update(round_num, test_loss, test_acc))

    # Log CSV (exact header)
    _ensure_header(CENTRAL_CSV, CENTRAL_HEADER)
    with open(CENTRAL_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            int(round_num),
            rec["central_train_loss"], rec["central_train_accuracy"],
            rec["central_test_loss"],  rec["central_test_accuracy"],
            rec["central_loss_gap"],   rec["central_accuracy_gap"],
            rec["conv_loss_rate"],     rec["conv_acc_rate"],
            rec["conv_loss_stability"],rec["conv_acc_stability"],
        ])

    # Return Flower evaluate tuple: (loss, metrics)
    return float(test_loss), rec


# (Removed legacy aggregate_and_log and aggregate_fit_metrics; using seeded variants defined inside server_fn)


# ───────────────────────────── Server app ─────────────────────────────
def server_fn(context: Context) -> ServerAppComponents:
    global strategy

    dataset_flag = context.node_config.get("dataset_flag", "cifar10")

    # Config
    num_rounds            = context.run_config["num-server-rounds"]
    fraction_fit          = context.run_config["fraction-fit"]
    min_available_clients = context.run_config.get("min_available_clients", 2)

    # pFedMe hyperparameters
    lamda       = context.run_config.get("lamda", 15.0)
    inner_steps = context.run_config.get("inner_steps", 5)
    outer_steps = context.run_config.get("outer_steps", 1)
    inner_lr    = context.run_config.get("inner_lr", 0.01)
    outer_lr    = context.run_config.get("outer_lr", 0.01)
    beta        = context.run_config.get("beta", 1.0)

    # Dirichlet α (required, used on client side; enforce presence here)
    try:
        dirichlet_alpha = context.run_config["dirichlet_alpha"]
        _ = float(dirichlet_alpha)  # just to ensure it's numeric
    except KeyError as e:
        raise KeyError(
            "Missing 'dirichlet_alpha' in [tool.flwr.app.config] of pyproject.toml"
        ) from e

    # Per-seed metrics directory and seeding
    seed = int(context.run_config.get("seed", 0))
    set_global_seed(seed)
    metrics_dir = os.path.join("metrics", f"seed_{seed}")
    os.makedirs(metrics_dir, exist_ok=True)
    global METRICS_DIR, CENTRAL_CSV, DIST_CSV, FIT_CSV
    METRICS_DIR = metrics_dir
    CENTRAL_CSV = os.path.join(metrics_dir, "centralized_metrics.csv")
    DIST_CSV    = os.path.join(metrics_dir, "distributed_metrics.csv")
    FIT_CSV     = os.path.join(metrics_dir, "fit_metrics.csv")

    # Central eval loaders cached once; we also use them to infer model shape
    central_bs = int(context.run_config.get("central_batch_size", 64))
    _ensure_central_cache(central_bs)
    in_ch, H, W = _CENTRAL_CACHE["shape"]
    num_classes = _CENTRAL_CACHE["num_classes"]

    # Initialize global parameters from a freshly-created model
    ndarrays   = get_weights(Net(in_ch=in_ch, img_h=H, img_w=W, n_class=num_classes))
    parameters = ndarrays_to_parameters(ndarrays)

    # Aggregators (mirror FedProx naming and dynamic CSV headers)
    def aggregate_fit_metrics_seeded(*args, **kwargs) -> Metrics:
        round_num = fit_round_counter["value"]
        CLIENT_CACHE.setdefault(round_num, {})

        # Normalize call styles to a list of (num_examples, metrics)
        metrics_list: List[Tuple[int, Metrics]] = []
        if len(args) == 3:
            _server_round, results, _failures = args
            for _, fitres in results:
                metrics_list.append((fitres.num_examples, fitres.metrics or {}))
        elif len(args) == 1:
            metrics_list = list(args[0])

        # Collect per-client for cache and aggregates
        comp_times   = []
        up_MB        = []
        dn_MB        = []
        train_losses = []
        train_accs   = []
        inner_batches = []
        train_samples = []

        for _, m in metrics_list:
            cid = int(m.get("cid", -1))
            if cid >= 0:
                CLIENT_CACHE[round_num][cid] = {
                    "train_loss_mean": float(m.get("train_loss_mean", 0.0)),
                    "train_accuracy_mean": float(m.get("train_accuracy_mean", 0.0)),
                    "upload_MB": float(m.get("upload_MB", 0.0)),
                    "download_MB": float(m.get("download_MB", 0.0)),
                    "comp_time_sec": float(m.get("comp_time_sec", 0.0)),
                    "inner_steps": int(m.get("inner_steps", 0)),
                    "inner_batches": float(m.get("inner_batches", 0.0)),
                    "total_train_samples": int(m.get("total_train_samples", 0)),
                }
            comp_times.append(float(m.get("comp_time_sec", 0.0)))
            up_MB.append(float(m.get("upload_MB", 0.0)))
            dn_MB.append(float(m.get("download_MB", 0.0)))
            train_losses.append(float(m.get("train_loss_mean", 0.0)))
            train_accs.append(float(m.get("train_accuracy_mean", 0.0)))
            inner_batches.append(float(m.get("inner_batches", 0.0)))
            train_samples.append(int(m.get("total_train_samples", 0)))

        result = {
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
            "avg_upload_MB": float(np.mean(up_MB)) if up_MB else 0.0,
            "total_upload_MB": float(np.sum(up_MB)) if up_MB else 0.0,
            "avg_download_MB": float(np.mean(dn_MB)) if dn_MB else 0.0,
            "total_download_MB": float(np.sum(dn_MB)) if dn_MB else 0.0,
            "avg_inner_batches": float(np.mean(inner_batches)) if inner_batches else 0.0,
            "total_train_samples": float(np.sum(train_samples)) if train_samples else 0.0,
        }

        path = os.path.join(metrics_dir, "fit_metrics.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            fieldnames = ["round"] + list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({"round": int(round_num), **result})

        fit_round_counter["value"] += 1
        return result

    def aggregate_and_log_seeded(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
        round_num = dist_round_counter["value"]

        # Weighted averages with example counts
        n_total = int(sum(n for n, _ in metrics_list))
        accs    = [float(m.get("test_accuracy", 0.0)) for _, m in metrics_list]
        losses  = [float(m.get("test_loss", 0.0))     for _, m in metrics_list]
        avg_acc  = float(sum(n * float(m.get("test_accuracy", 0.0)) for n, m in metrics_list) / max(1, n_total))
        avg_loss = float(sum(n * float(m.get("test_loss", 0.0))      for n, m in metrics_list) / max(1, n_total))
        acc_sd  = float(np.std(accs))   if accs else 0.0
        loss_sd = float(np.std(losses)) if losses else 0.0
        n = max(1, len(accs))
        acc_se  = acc_sd  / (n ** 0.5)
        loss_se = loss_sd / (n ** 0.5)

        # Computation/communication/wallclock
        comp_times = [float(m.get("comp_time_sec", 0.0)) for _, m in metrics_list]
        up_MB_list, dn_MB_list = [], []
        cache = CLIENT_CACHE.get(round_num, {})
        for _, m in metrics_list:
            cid = int(m.get("cid", m.get("client_id", -1)))
            fit_m = cache.get(cid, {})
            up_MB_list.append(float(m.get("upload_MB", fit_m.get("upload_MB", 0.0))))
            dn_MB_list.append(float(m.get("download_MB", fit_m.get("download_MB", 0.0))))

        network_speed_mbps = 10.0
        # MB (megabytes) to Mb (megabits) requires ×8 for time in seconds at Mb/s
        comm_times = [8.0 * (up + dn) / network_speed_mbps for up, dn in zip(up_MB_list, dn_MB_list)]
        wall_clock_times = [c + comm for c, comm in zip(comp_times, comm_times)]

        result: Dict[str, Any] = {
            "avg_accuracy":   avg_acc,
            "avg_loss":       avg_loss,
            "accuracy_std":   acc_sd,
            "loss_std":       loss_sd,
            "acc_ci95_lo":    float(avg_acc  - 1.96 * acc_se),
            "acc_ci95_hi":    float(avg_acc  + 1.96 * acc_se),
            "loss_ci95_lo":   float(avg_loss - 1.96 * loss_se),
            "loss_ci95_hi":   float(avg_loss + 1.96 * loss_se),
            "avg_comp_time_sec":   float(np.mean(comp_times)) if comp_times else 0.0,
            "total_comp_time_sec": float(np.sum(comp_times))  if comp_times else 0.0,
            "std_comp_time_sec":   float(np.std(comp_times)) if comp_times else 0.0,
            "avg_upload_MB":       float(np.mean(up_MB_list)) if up_MB_list else 0.0,
            "total_upload_MB":     float(np.sum(up_MB_list)) if up_MB_list else 0.0,
            "avg_download_MB":     float(np.mean(dn_MB_list)) if dn_MB_list else 0.0,
            "total_download_MB":   float(np.sum(dn_MB_list)) if dn_MB_list else 0.0,
            "total_communication_MB": float((np.sum(up_MB_list) + np.sum(dn_MB_list))) if (up_MB_list and dn_MB_list) else 0.0,
            "avg_comm_time_sec":    float(np.mean(comm_times)) if comm_times else 0.0,
            "total_comm_time_sec":  float(np.sum(comm_times)) if comm_times else 0.0,
            "avg_wall_clock_sec":   float(np.mean(wall_clock_times)) if wall_clock_times else 0.0,
            "total_wall_clock_sec": float(np.sum(wall_clock_times)) if wall_clock_times else 0.0,
            "std_wall_clock_sec":   float(np.std(wall_clock_times)) if wall_clock_times else 0.0,
        }

        for idx, (_, m) in enumerate(metrics_list):
            result[f"client_{idx + 1}_accuracy"] = float(m.get("test_accuracy", 0.0))
            result[f"client_{idx + 1}_loss"]     = float(m.get("test_loss", 0.0))

        csv_path = os.path.join(metrics_dir, "distributed_metrics.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            fieldnames = ["round"] + list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({"round": int(round_num), **result})

        dist_round_counter["value"] += 1
        return result

    # Strategy
    strategy = pFedMe(
        lamda=lamda,
        inner_steps=inner_steps,
        outer_steps=outer_steps,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        beta=beta,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics_seeded,
        evaluate_metrics_aggregation_fn=aggregate_and_log_seeded,
        evaluate_fn=lambda rnd, params, cfg: evaluate_and_log_central(
            rnd, params, {**cfg, "dataset_flag": dataset_flag, "central_batch_size": central_bs}
        ),
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
