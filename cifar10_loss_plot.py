#!/usr/bin/env python3
"""
CIFAR-10 Federated Learning — Loss curves (styled like the template).

- Consistent colors/linestyles with the accuracy figure
- Bold axis labels and title
- Boxed legend on the right (outside), vertically centered
- Clear warnings if files/columns are missing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
METRICS_DIR = "cifar10-metrics"
PLOTS_DIR   = "plots"
OUTPUT_FILE = "cifar10_loss_comparison.png"

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

# Colors/linestyles aligned with the template & your accuracy plot
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
    "HierFL":    "-",  # dashed
    "CFL":       "-",   # dotted
    "Fedge-v1": "-",
}
LW = 2.4

# ---------- Input files & columns ----------
METHODS = {
    "Scaffold":  {"file": "scaffold_centralized_metrics.csv",
                  "x": "round", "loss": "central_test_loss"},
    "FedProx":   {"file": "fedprox_centralized_metrics.csv",
                  "x": "round", "loss": "central_test_loss"},
    "pFedMe":    {"file": "pfedme_centralized_metrics.csv",
                  "x": "round", "loss": "central_test_loss"},
    "HierFL":    {"file": "hierfl_global_model_metrics.csv",
                  "x": "global_round", "loss": "global_test_loss_centralized"},
    "CFL":       {"file": "cfl_clusters_metrics.csv",
                  "x": "round", "loss": "test_loss_mean"},
    "Fedge-v1": {"file": "fedge_cloud_cloud_round_metrics.csv",
                  "x": "global_round", "loss": "cluster_loss_mean"},
}

# ---------- Helpers ----------
def load_xy(method: str):
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

    xcol = cfg["x"]; ycol = cfg["loss"]
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

def style_axes(ax):
    ax.set_xlabel("Global Rounds")
    ax.set_ylabel("Test Loss")
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.8)
    ax.set_xlim(0, 150)
    ax.set_xticks([1] + list(range(10, 151, 10)))  # 1, 10, 20, ..., 150

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
    name = "HierFL"
    cfg = METHODS[name]
    fpath = os.path.join(METRICS_DIR, cfg["file"])
    expected = [cfg["x"], cfg["loss"]]
    if not os.path.exists(fpath):
        print(f"[warn] HierFL file not found: {fpath}")
        return
    df = pd.read_csv(fpath, nrows=2)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[warn] HierFL missing columns {missing}. Found: {list(df.columns)}")
    else:
        print(f"[ok] HierFL file/columns look good: {fpath}")

# ---------- Plot ----------
def create_loss_plot():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots()

    any_data = False
    for m in METHODS:
        x, y = load_xy(m)
        if x is None:
            continue
        ax.plot(x, y, LSTYLE[m], color=COL[m], linewidth=LW, label=m)
        any_data = True

    if not any_data:
        print("[error] No valid data found for any method!")
        return

    ax.set_title("Loss vs Rounds")
    style_axes(ax)
    add_legend(ax)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, OUTPUT_FILE)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[ok] Loss plot saved to: {out}")

def print_summary_statistics():
    print("\n" + "="*60)
    print("CIFAR-10 FEDERATED LEARNING LOSS SUMMARY")
    print("="*60)
    for m in METHODS:
        x, y = load_xy(m)
        if x is None:
            print(f"{m:12} | No data available")
            continue
        final_loss = float(y[-1])
        best = float(np.min(y))
        print(f"{m:12} | Final: {final_loss:6.4f} | Best: {best:6.4f} | Rounds: {len(x):3d}")
    print("="*60)

# ---------- Main ----------
if __name__ == "__main__":
    if not os.path.isdir(METRICS_DIR):
        raise SystemExit(f"[error] Metrics directory not found: {METRICS_DIR}")
    verify_hierfl()
    create_loss_plot()
    print_summary_statistics()
