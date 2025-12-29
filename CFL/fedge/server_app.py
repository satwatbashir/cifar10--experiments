"""fedge: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# Removed unused imports: FedAvg, FedProx
from fedge.task import Net, get_weights, load_data, load_global_data, set_weights, test, set_global_seed
from typing import List, Tuple
import os, csv
from statistics import mean, pstdev
import numpy as np
import torch
from fedge.task import DATA_FLAGS
from .cfl import CFL
# from .scaffold import Scaffold  # unused

# Per-seed metrics directory (set in server_fn)
METRICS_DIR = "metrics"

def _metrics_dir(context: Context) -> str:
    seed = int(context.run_config.get("seed", 0))
    d = os.path.join("metrics", f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    return d

strategy: CFL  # will be set in server_fn below

# For tracking distributed metrics rounds (evaluate)
dist_round_counter = {"value": 1}
# For tracking fit metrics rounds
fit_round_counter = {"value": 1}
# Cache of last fit metrics by CID for rollups
last_fit: dict[str, dict] = {}

# Strict schema enforcement during development
def require(m: Metrics, k: str):
    if k not in m:
        raise KeyError(f"Missing '{k}' in client metrics for this round")
    return m[k]

def _vectorize_control(ctrl: List[torch.Tensor]) -> torch.Tensor:
    """
    Safely flatten and concatenate a list of control‐variate tensors.
    Uses torch.flatten to avoid dtype/view issues.
    """
    flat_tensors = [torch.flatten(c) for c in ctrl]
    return torch.cat(flat_tensors, dim=0)

def _representative_weights_from_clusters(strategy, last_fit: dict[str, dict] | dict) -> list[torch.Tensor] | None:
    """Build a representative global model as a weighted average of cluster models.

    Weight of cluster c is sum of num_train_samples for its member clients (from last_fit).
    Fallback to cluster size if cache missing.
    Returns a list of tensors or None if unavailable.
    """
    if not getattr(strategy, "cluster_models", None):
        return None
    rep: list[torch.Tensor] | None = None
    total_w = 0.0
    for c, members in getattr(strategy, "clusters", {}).items():
        w_c = 0
        for cid in members:
            w_c += int(last_fit.get(str(cid), {}).get("num_train_samples", 0))
        if w_c <= 0:
            w_c = len(members)
        model = strategy.cluster_models.get(c)
        if model is None:
            continue
        if rep is None:
            rep = [t.detach().clone().float() * float(w_c) for t in model]
        else:
            for i, t in enumerate(model):
                rep[i] += t.detach().float() * float(w_c)
        total_w += float(w_c)
    if rep is None or total_w <= 0:
        return None
    for i in range(len(rep)):
        rep[i] /= total_w
    return rep

def aggregate_evaluate_metrics(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate per-round distributed metrics and log centralized evaluation.

    - Writes METRICS_DIR/distributed_metrics.csv with averages/std/CI.
    - Evaluates a representative cluster model on the global test set and writes
      METRICS_DIR/centralized_metrics.csv including convergence fields.
    """
    global dist_round_counter, strategy, METRICS_DIR

    round_num = dist_round_counter["value"]

    # Build cid -> metrics
    by_cid: dict[str, Metrics] = {}
    for _num, m in metrics_list:
        cid = str(require(m, "cid"))
        _ = int(require(m, "cluster_id"))
        by_cid[cid] = m

    # Distributed rollup (across all participating clients)
    n = len(by_cid)
    tacc  = [float(require(m, "test_accuracy"))  for m in by_cid.values()]
    tloss = [float(require(m, "test_loss"))      for m in by_cid.values()]
    tracc = [float(require(m, "train_accuracy")) for m in by_cid.values()]
    trlos = [float(require(m, "train_loss"))     for m in by_cid.values()]
    agap  = [float(require(m, "accuracy_gap"))   for m in by_cid.values()]
    lgap  = [float(require(m, "loss_gap"))       for m in by_cid.values()]
    dne   = [int(require(m, "download_bytes_eval")) for m in by_cid.values()]
    upe   = [int(require(m, "upload_bytes_eval"))   for m in by_cid.values()]
    # Combine with last fit cache for these clients
    dnf = [int(last_fit.get(cid, {}).get("download_bytes_fit", 0)) for cid in by_cid.keys()]
    upf = [int(last_fit.get(cid, {}).get("upload_bytes_fit", 0))   for cid in by_cid.keys()]
    ctf = [float(last_fit.get(cid, {}).get("comp_time_fit_sec", last_fit.get(cid, {}).get("comp_time_sec", 0.0))) for cid in by_cid.keys()]

    def _mean(xs):
        return float(mean(xs)) if xs else 0.0
    def _std(xs):
        return float(pstdev(xs)) if len(xs) > 1 else 0.0
    def _ci95(xs):
        return 1.96 * _std(xs) / (np.sqrt(len(xs)) if len(xs) > 0 else 1.0)

    os.makedirs(METRICS_DIR, exist_ok=True)
    dist_path = os.path.join(METRICS_DIR, "distributed_metrics.csv")
    dist_fields = [
        "round","num_clients",
        "test_accuracy_mean","test_accuracy_std","test_accuracy_ci95",
        "test_accuracy_ci95_lo","test_accuracy_ci95_hi",
        "test_loss_mean","test_loss_std","test_loss_ci95",
        "test_loss_ci95_lo","test_loss_ci95_hi",
        "train_accuracy_mean","train_accuracy_std",
        "train_loss_mean","train_loss_std",
        "accuracy_gap_mean","loss_gap_mean",
        "download_MB_eval_sum","upload_MB_eval_sum",
        "download_MB_fit_sum","upload_MB_fit_sum",
        "comp_time_fit_sec_sum",
        "download_MB_fit_avg","download_MB_fit_std",
        "upload_MB_fit_avg","upload_MB_fit_std",
        "comp_time_fit_sec_avg","comp_time_fit_sec_std",
        "total_communication_MB",
        # Aliases for legacy parsers (FedProx naming)
        "avg_accuracy","accuracy_std","avg_loss","loss_std",
    ]
    write_header = not os.path.exists(dist_path)
    fit_dn_MB = [b / (1024 * 1024) for b in dnf]
    fit_up_MB = [b / (1024 * 1024) for b in upf]
    comp_secs = ctf
    total_comm_MB = (sum(dnf) + sum(upf)) / (1024 * 1024)
    # CI lo/hi for accuracy and loss (explicit bounds)
    m_acc, s_acc = _mean(tacc), _std(tacc)
    ci_acc = 1.96 * s_acc / (np.sqrt(len(tacc)) if len(tacc) > 0 else 1.0)
    acc_ci95_lo, acc_ci95_hi = m_acc - ci_acc, m_acc + ci_acc
    m_loss, s_loss = _mean(tloss), _std(tloss)
    ci_l = 1.96 * s_loss / (np.sqrt(len(tloss)) if len(tloss) > 0 else 1.0)
    loss_ci95_lo, loss_ci95_hi = m_loss - ci_l, m_loss + ci_l

    with open(dist_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dist_fields)
        if write_header:
            w.writeheader()
        w.writerow({
            "round": round_num,
            "num_clients": n,
            "test_accuracy_mean": m_acc,
            "test_accuracy_std": s_acc,
            "test_accuracy_ci95": _ci95(tacc),
            "test_accuracy_ci95_lo": acc_ci95_lo,
            "test_accuracy_ci95_hi": acc_ci95_hi,
            "test_loss_mean": m_loss,
            "test_loss_std": s_loss,
            "test_loss_ci95": _ci95(tloss),
            "test_loss_ci95_lo": loss_ci95_lo,
            "test_loss_ci95_hi": loss_ci95_hi,
            "train_accuracy_mean": _mean(tracc),
            "train_accuracy_std": _std(tracc),
            "train_loss_mean": _mean(trlos),
            "train_loss_std": _std(trlos),
            "accuracy_gap_mean": _mean(agap),
            "loss_gap_mean": _mean(lgap),
            "download_MB_eval_sum": sum(dne) / (1024 * 1024),
            "upload_MB_eval_sum":   sum(upe) / (1024 * 1024),
            "download_MB_fit_sum":  sum(dnf) / (1024 * 1024),
            "upload_MB_fit_sum":    sum(upf) / (1024 * 1024),
            "comp_time_fit_sec_sum": sum(comp_secs),
            "download_MB_fit_avg": _mean(fit_dn_MB),
            "download_MB_fit_std": _std(fit_dn_MB),
            "upload_MB_fit_avg":   _mean(fit_up_MB),
            "upload_MB_fit_std":   _std(fit_up_MB),
            "comp_time_fit_sec_avg": _mean(comp_secs),
            "comp_time_fit_sec_std": _std(comp_secs),
            "total_communication_MB": total_comm_MB,
            # Aliases for legacy parsers (FedProx naming)
            "avg_accuracy": m_acc,
            "accuracy_std": s_acc,
            "avg_loss": m_loss,
            "loss_std": s_loss,
        })

    # Optional detailed per-client eval rows for debugging/parity
    try:
        clients_path = os.path.join(METRICS_DIR, "clients_metrics.csv")
        clients_fields = [
            "round","cid","cluster_id",
            "test_loss","test_accuracy","train_loss","train_accuracy",
            "download_MB_eval","upload_MB_eval",
        ]
        write_clients_header = not os.path.exists(clients_path)
        with open(clients_path, "a", newline="") as cf:
            wcl = csv.DictWriter(cf, fieldnames=clients_fields)
            if write_clients_header:
                wcl.writeheader()
            for cid, m in sorted(by_cid.items()):
                wcl.writerow({
                    "round": round_num,
                    "cid": cid,
                    "cluster_id": int(require(m, "cluster_id")),
                    "test_loss": float(require(m, "test_loss")),
                    "test_accuracy": float(require(m, "test_accuracy")),
                    "train_loss": float(require(m, "train_loss")),
                    "train_accuracy": float(require(m, "train_accuracy")),
                    "download_MB_eval": int(require(m, "download_bytes_eval")) / (1024 * 1024),
                    "upload_MB_eval":   int(require(m, "upload_bytes_eval"))   / (1024 * 1024),
                })
    except Exception as e:
        print(f"[warn] writing clients_metrics.csv failed: {e}")

    # Centralized evaluation: evaluate representative model on global train and test sets
    try:
        model_tensors = _representative_weights_from_clusters(strategy, last_fit)
        if model_tensors is not None:
            net = Net(in_ch=3, n_class=10)
            nd = [t.detach().cpu().numpy() for t in model_tensors]
            set_weights(net, nd)
            trainloader, testloader, _ = load_global_data(batch_size=256)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_loss, train_acc = test(net, trainloader, device)
            test_loss,  test_acc  = test(net, testloader,  device)
            # Convergence rates based on test metrics
            extra = ctracker.update(round_num, test_loss, test_acc)
            cent_path = os.path.join(METRICS_DIR, "centralized_metrics.csv")
            cent_fields = [
                "round",
                "central_train_loss","central_train_accuracy",
                "central_test_loss","central_test_accuracy",
                "central_loss_gap","central_accuracy_gap",
                "conv_loss_rate","conv_acc_rate","conv_loss_stability","conv_acc_stability",
            ]
            write_header_c = not os.path.exists(cent_path)
            with open(cent_path, "a", newline="") as cf:
                w2 = csv.DictWriter(cf, fieldnames=cent_fields)
                if write_header_c:
                    w2.writeheader()
                w2.writerow({
                    "round": round_num,
                    "central_train_loss": train_loss,
                    "central_train_accuracy": train_acc,
                    "central_test_loss": test_loss,
                    "central_test_accuracy": test_acc,
                    "central_loss_gap": test_loss - train_loss,
                    "central_accuracy_gap": train_acc - test_acc,
                    "conv_loss_rate":      extra.get("conv_loss_rate", ""),
                    "conv_acc_rate":       extra.get("conv_acc_rate", ""),
                    "conv_loss_stability": extra.get("conv_loss_stability", ""),
                    "conv_acc_stability":  extra.get("conv_acc_stability", ""),
                })
    except Exception as e:
        print(f"[warn] centralized eval failed: {e}")

    dist_round_counter["value"] += 1

    # Return aggregate
    mean_acc = float(_mean(tacc)) if by_cid else 0.0
    mean_loss = float(_mean(tloss)) if by_cid else 0.0
    return {"avg_test_accuracy": mean_acc, "avg_test_loss": mean_loss, "round": round_num}


def aggregate_fit_metrics(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """Cache last per-CID fit metrics and write per-round aggregated CSV.

    Flower passes results as List[(num_train_samples, metrics_dict)]. We must
    extract the client ID from metrics_dict["cid"].
    """
    global fit_round_counter, METRICS_DIR

    round_num = fit_round_counter["value"]
    # Cache last fit metrics by CID and collect for aggregation
    trained = 0
    train_losses, train_accs = [], []
    dnbytes, upbytes, compsecs = [], [], []
    inner_batches, num_train_samples = [], []
    for num_train, m in metrics_list:
        cid = str(require(m, "cid"))
        _ = int(require(m, "cluster_id"))
        tl = float(m.get("train_loss_mean", 0.0))
        ta = float(m.get("train_accuracy_mean", 0.0))
        dl = int(require(m, "download_bytes_fit"))
        ul = int(require(m, "upload_bytes_fit"))
        ct = float(m.get("comp_time_fit_sec", m.get("comp_time_sec", 0.0)))
        ib = int(m.get("inner_batches", 0))
        nts = int(m.get("num_train_samples", num_train))
        last_fit[cid] = {
            "download_bytes_fit": dl,
            "upload_bytes_fit":   ul,
            "comp_time_fit_sec":  ct,
            "num_train_samples":  nts,
            "cluster_id":         int(m.get("cluster_id")),
        }
        trained += 1
        train_losses.append(tl)
        train_accs.append(ta)
        dnbytes.append(dl)
        upbytes.append(ul)
        compsecs.append(ct)
        inner_batches.append(ib)
        num_train_samples.append(nts)

    def _mean(xs):
        return float(mean(xs)) if xs else 0.0
    def _std(xs):
        return float(pstdev(xs)) if len(xs) > 1 else 0.0

    os.makedirs(METRICS_DIR, exist_ok=True)
    fit_path = os.path.join(METRICS_DIR, "fit_metrics.csv")
    fit_fields = [
        "round","num_clients",
        "train_loss_mean","train_loss_std","train_loss_min","train_loss_max",
        "train_accuracy_mean","train_accuracy_std","train_accuracy_min","train_accuracy_max",
        "download_MB_sum","download_MB_avg","download_MB_std",
        "upload_MB_sum","upload_MB_avg","upload_MB_std",
        "comp_time_sec_sum","comp_time_sec_avg","comp_time_sec_std",
        "inner_batches_sum","inner_batches_avg","inner_batches_std",
        "num_train_samples_sum",
    ]
    write_header = not os.path.exists(fit_path)
    dn_MB = [b / (1024 * 1024) for b in dnbytes]
    up_MB = [b / (1024 * 1024) for b in upbytes]

    with open(fit_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fit_fields)
        if write_header:
            w.writeheader()
        w.writerow({
            "round": round_num,
            "num_clients": trained,
            "train_loss_mean": _mean(train_losses),
            "train_loss_std": _std(train_losses),
            "train_loss_min": (min(train_losses) if train_losses else 0.0),
            "train_loss_max": (max(train_losses) if train_losses else 0.0),
            "train_accuracy_mean": _mean(train_accs),
            "train_accuracy_std": _std(train_accs),
            "train_accuracy_min": (min(train_accs) if train_accs else 0.0),
            "train_accuracy_max": (max(train_accs) if train_accs else 0.0),
            "download_MB_sum": sum(dn_MB),
            "download_MB_avg": _mean(dn_MB),
            "download_MB_std": _std(dn_MB),
            "upload_MB_sum":   sum(up_MB),
            "upload_MB_avg":   _mean(up_MB),
            "upload_MB_std":   _std(up_MB),
            "comp_time_sec_sum": sum(compsecs),
            "comp_time_sec_avg": _mean(compsecs),
            "comp_time_sec_std": _std(compsecs),
            "inner_batches_sum": int(sum(inner_batches)),
            "inner_batches_avg": _mean(inner_batches),
            "inner_batches_std": _std(inner_batches),
            "num_train_samples_sum": int(sum(num_train_samples)),
        })

    fit_round_counter["value"] += 1
    return {"trained_clients": trained, "round": round_num}

# Centralized convergence tracker
class ConvergenceTracker:
    def __init__(self):
        self.prev_loss = None
        self.prev_acc = None
        self.loss_changes = []
        self.acc_changes  = []
    def update(self, round_num, loss, acc) -> dict:
        if self.prev_loss is None or round_num == 0:
            self.prev_loss, self.prev_acc = loss, acc
            return {}
        dl = loss - self.prev_loss
        da = acc  - self.prev_acc
        self.loss_changes.append(dl)
        self.acc_changes.append(da)
        self.prev_loss, self.prev_acc = loss, acc
        return {
            "conv_loss_rate":      float(dl),
            "conv_acc_rate":       float(da),
            "conv_loss_stability": float(np.var(self.loss_changes)),
            "conv_acc_stability":  float(np.var(self.acc_changes)),
        }
ctracker = ConvergenceTracker()
dataset_flag = "cifar10"


# Note: Centralized evaluation is logged inside aggregate_evaluate_metrics

# Cluster metrics logger
def server_fn(context: Context):
    global strategy, METRICS_DIR
    dataset_flag = context.node_config.get("dataset_flag", "cifar10")

    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    fraction_evaluate = float(context.run_config.get("fraction_evaluate", 1.0))
    min_available_clients = int(context.run_config.get("min_available_clients", 2))
    seed = int(context.run_config.get("seed", 0))
    personalize_eval = bool(context.run_config.get("personalize_eval", False))

    # Seed RNGs per server process
    set_global_seed(seed)

    # Compute per-seed metrics dir
    METRICS_DIR = _metrics_dir(context)

    trainloader, valloader, num_classes = load_data(dataset_flag, partition_id=0, num_partitions=1, batch_size=1,
        dirichlet_alpha=float(context.run_config.get("dirichlet_alpha", 0.5)), base_seed=seed)

    sample, _ = next(iter(trainloader))
    if isinstance(sample, torch.Tensor):
        _, in_ch, H, W = sample.shape
    else:
        print("Sample is NOT a tensor! Type:", type(sample))
        print("Sample content:", sample)
        raise ValueError("Sample is not a tensor. Did your transform apply?")

    ndarrays   = get_weights(Net(in_ch=in_ch, n_class=num_classes))
    parameters = ndarrays_to_parameters(ndarrays)


    strategy = CFL(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=1,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        # CFL-specific parameters

        eps_1=0.8,
        eps_2=0.8,
        min_cluster_size=2,
        gamma_max=0.15,
        strict_participation=False,
        metrics_dir=METRICS_DIR,
        personalize_eval=personalize_eval,
        verbose=False,
        # Callbacks
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        # evaluate_fn removed for strict CFL (no centralized eval)
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
