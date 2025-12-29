#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
METRICS_DIR = "cifar10-metrics"
PLOTS_DIR   = "plots"
ACC_FNAME   = "cifar10_accuracy_comparison.png"
LOSS_FNAME  = "cifar10_loss_comparison.png"

# ---------- Visual style ----------
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

COL = {
    "Scaffold":  "#1f77b4",  # blue
    "FedProx":   "#ff7f0e",  # orange
    "pFedMe":    "#2ca02c",  # green
    "HierFL":    "#d62728",  # red
    "CFL":       "#9467bd",  # purple
    "Fedge-v1": "#8c564b",  # brown
}

LSTYLE = {
    "Scaffold":  "-",
    "FedProx":   "-",
    "pFedMe":    "-",
    "HierFL":    "-",  # dashed like the template
    "CFL":       "-",   # dotted like the template
    "Fedge-v1": "-",
}

LW = 2.4

# ---------- Input files & columns ----------
METHODS = {
    "Scaffold":  {"file": "scaffold_centralized_metrics.csv",
                  "x": "round", "acc": "central_test_accuracy", "loss": "central_test_loss"},
    "FedProx":   {"file": "fedprox_centralized_metrics.csv",
                  "x": "round", "acc": "central_test_accuracy", "loss": "central_test_loss"},
    "pFedMe":    {"file": "pfedme_centralized_metrics.csv",
                  "x": "round", "acc": "central_test_accuracy", "loss": "central_test_loss"},
    "HierFL":    {"file": "hierfl_global_model_metrics.csv",
                  "x": "global_round", "acc": "global_test_accuracy_centralized", "loss": "global_test_loss_centralized"},
    "CFL":       {"file": "cfl_clusters_metrics.csv",
                  "x": "round", "acc": "test_acc_mean", "loss": "test_loss_mean"},
    "Fedge-v1": {"file": "fedge_cloud_cloud_round_metrics.csv",
                  "x": "global_round", "acc": "cluster_accuracy_mean", "loss": "cluster_loss_mean"},
}

# ---------- Helpers ----------
def load_xy(method: str, col_key: str):
    cfg = METHODS[method]
    fpath = os.path.join(METRICS_DIR, cfg["file"])
    if not os.path.exists(fpath):
        print(f"[warn] Missing file for {method}: {fpath}")
        return None, None

    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"[warn] Could not read {fpath}: {e}")
        return None, None

    xcol = cfg["x"]; ycol = cfg[col_key]
    missing = [c for c in (xcol, ycol) if c not in df.columns]
    if missing:
        print(f"[warn] {method}: missing columns {missing}. Have: {list(df.columns)}")
        return None, None

    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        print(f"[warn] {method}: no valid data after NaN filtering.")
        return None, None
    return x, y

def style_axes(ax, y_label, y_lim=None, y_ticks=None, x_lim=(0, 150), x_ticks=None):
    ax.set_xlabel("Global Rounds")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.8)
    if x_lim:   ax.set_xlim(x_lim)
    if x_ticks: ax.set_xticks(x_ticks)
    if y_lim:   ax.set_ylim(y_lim)
    if y_ticks: ax.set_yticks(y_ticks)

def add_legend(ax):
    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="lightgray",
        borderpad=0.8,
        ncol=1,
    )
    for legline in leg.get_lines():
        legline.set_linewidth(2.6)

def verify_hierfl():
    """Explicit check for HierFL presence and schema."""
    name = "HierFL"
    cfg = METHODS[name]
    fpath = os.path.join(METRICS_DIR, cfg["file"])
    expected = [cfg["x"], cfg["acc"], cfg["loss"]]
    if not os.path.exists(fpath):
        print(f"[warn] HierFL file not found: {fpath}")
        return
    df = pd.read_csv(fpath, nrows=2)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[warn] HierFL missing columns {missing}. Found: {list(df.columns)}")
    else:
        print(f"[ok] HierFL file/columns look good: {fpath}")

# ---------- Plotters ----------
def plot_accuracy():
    fig, ax = plt.subplots()
    any_data = False
    for m in METHODS:
        x, y = load_xy(m, "acc")
        if x is None: 
            continue
        ax.plot(x, y, LSTYLE[m], color=COL[m], linewidth=LW, label=m)
        any_data = True
    if not any_data:
        print("[error] No accuracy data found."); return
    ax.set_title("Accuracy vs Rounds")
    style_axes(ax, "Test Accuracy",
               y_lim=(0.0, 1.0),
               y_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               x_lim=(0, 150),
               x_ticks=[1] + list(range(10, 151, 10)))
    add_legend(ax)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, ACC_FNAME), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[ok] Saved accuracy figure: {os.path.join(PLOTS_DIR, ACC_FNAME)}")

def plot_loss():
    fig, ax = plt.subplots()
    any_data = False
    for m in METHODS:
        x, y = load_xy(m, "loss")
        if x is None:
            continue
        ax.plot(x, y, LSTYLE[m], color=COL[m], linewidth=LW, label=m)
        any_data = True
    if not any_data:
        print("[error] No loss data found."); return
    ax.set_title("Loss vs Rounds")
    style_axes(ax, "Test Loss",
               x_lim=(0, 150),
               x_ticks=[1] + list(range(10, 151, 10)))
    add_legend(ax)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, LOSS_FNAME), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[ok] Saved loss figure: {os.path.join(PLOTS_DIR, LOSS_FNAME)}")

# ---------- Main ----------
if __name__ == "__main__":
    if not os.path.isdir(METRICS_DIR):
        raise SystemExit(f"[error] Metrics directory not found: {METRICS_DIR}")
    verify_hierfl()   # explicit check
    plot_accuracy()
    plot_loss()
