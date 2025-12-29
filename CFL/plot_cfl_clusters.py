"""Generate CFL metrics plots and summaries.

This script produces:
1) Main figure (2x2):
   - Top-left: Per-cluster test accuracy with ±1 std fill and overall mean across all clients (overlay)
   - Top-right: Per-cluster test loss with ±1 std fill
   - Bottom-left: Total communication cost per round (MB)
   - Bottom-right: Wall clock time per round with number of clusters as secondary axis
2) Per-cluster small multiples for accuracy and loss
3) System overview plots
4) CSV exports to plots/ for:
   - overall_mean_test_accuracy_by_round.csv
   - per_cluster_test_accuracy_summary.csv
   - overview_series.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "metrics"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def read_csv(name: str) -> pd.DataFrame:
    """Read CSV from metrics directory."""
    path = METRICS_DIR / name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return pd.DataFrame()

# Load all data
clients_df = read_csv("clients_metrics.csv")
clusters_df = read_csv("clusters_metrics.csv")
system_df = read_csv("system_metrics.csv")
clusters_over_time_df = read_csv("clusters_over_time.csv")

# Fallback: derive cluster rollups from clients_df if clusters_df is missing/empty
if (clusters_df is None) or clusters_df.empty:
    if clients_df is not None and not clients_df.empty:
        print("clusters_metrics.csv missing; deriving cluster rollups from clients_metrics.csv")
        df = clients_df.copy()
        # Ensure expected columns exist
        for col in [
            "download_bytes_eval","upload_bytes_eval",
            "download_bytes_fit","upload_bytes_fit",
            "comp_time_fit_sec",
        ]:
            if col not in df.columns:
                df[col] = 0.0
        # Bytes -> MB
        to_mb = lambda x: float(x) / (1024 * 1024)
        df["download_MB_eval_sum"] = df["download_bytes_eval"].apply(to_mb)
        df["upload_MB_eval_sum"]   = df["upload_bytes_eval"].apply(to_mb)
        df["download_MB_fit_sum"]  = df["download_bytes_fit"].apply(to_mb)
        df["upload_MB_fit_sum"]    = df["upload_bytes_fit"].apply(to_mb)
        df["comp_time_fit_sec_sum"] = df["comp_time_fit_sec"].astype(float)
        # Aggregate
        grp = df.groupby(["round","cluster_id"], as_index=False)
        clusters_df = grp.agg({
            "test_acc": "mean",
            "test_loss": "mean",
            "train_acc": "mean",
            "train_loss": "mean",
            "acc_gap": "mean",
            "loss_gap": "mean",
            "download_MB_eval_sum": "sum",
            "upload_MB_eval_sum":   "sum",
            "download_MB_fit_sum":  "sum",
            "upload_MB_fit_sum":    "sum",
            "comp_time_fit_sec_sum": "sum",
            "cid": "count",
        }).rename(columns={
            "test_acc": "test_acc_mean",
            "test_loss": "test_loss_mean",
            "train_acc": "train_acc_mean",
            "train_loss": "train_loss_mean",
            "acc_gap": "acc_gap_mean",
            "loss_gap": "loss_gap_mean",
            "cid": "size",
        })

# Build overview series for CSV exports and plotting
overview_df = pd.DataFrame()
if not clusters_df.empty:
    macro_acc = (
        clusters_df.groupby("round")["test_acc_mean"].mean().reset_index()
        .rename(columns={"test_acc_mean": "macro_cluster_test_acc"})
    )
    macro_gap = (
        clusters_df.assign(gap=lambda d: d["train_acc_mean"] - d["test_acc_mean"])  # train - test
        .groupby("round")["gap"].mean().reset_index()
        .rename(columns={"gap": "macro_cluster_gap"})
    )
    overview_df = pd.merge(macro_acc, macro_gap, on="round", how="outer")
else:
    overview_df = pd.DataFrame(columns=["round","macro_cluster_test_acc","macro_cluster_gap"])

if not clients_df.empty:
    avg_client_acc = (
        clients_df.groupby("round")["test_acc"].mean().reset_index()
        .rename(columns={"test_acc": "avg_client_test_acc"})
    )
    avg_client_loss = (
        clients_df.groupby("round")["test_loss"].mean().reset_index()
        .rename(columns={"test_loss": "avg_client_test_loss"})
    )
else:
    if not clusters_df.empty and {"size","test_acc_mean","test_loss_mean"}.issubset(clusters_df.columns):
        tmp = clusters_df[["round","size","test_acc_mean","test_loss_mean"]].copy()
        w = tmp.groupby("round").apply(
            lambda df: pd.Series({
                "avg_client_test_acc": (df["size"] * df["test_acc_mean"]).sum() / max(df["size"].sum(), 1),
                "avg_client_test_loss": (df["size"] * df["test_loss_mean"]).sum() / max(df["size"].sum(), 1),
            })
        ).reset_index()
    else:
        w = pd.DataFrame(columns=["round","avg_client_test_acc","avg_client_test_loss"])
    avg_client_acc = w[["round","avg_client_test_acc"]]
    avg_client_loss = w[["round","avg_client_test_loss"]]

overview_df = (
    overview_df.merge(avg_client_acc, on="round", how="outer")
               .merge(avg_client_loss, on="round", how="outer")
               .sort_values("round")
)

# CSV exports
try:
    # overall mean accuracy by round (client-averaged)
    (overview_df[["round","avg_client_test_acc"]]
     .rename(columns={"avg_client_test_acc": "test_acc"})
     .to_csv(PLOTS_DIR / "overall_mean_test_accuracy_by_round.csv", index=False))
except Exception as e:
    print(f"Export error (overall mean acc): {e}")

try:
    if not clusters_df.empty and "test_acc_mean" in clusters_df.columns:
        def _summ(cdf: pd.DataFrame) -> pd.Series:
            cdf = cdf.sort_values("round")
            best_idx = cdf["test_acc_mean"].idxmax()
            return pd.Series({
                "final_round": int(cdf["round"].iloc[-1]),
                "final_acc": float(cdf["test_acc_mean"].iloc[-1]),
                "best_acc": float(cdf.loc[best_idx, "test_acc_mean"]),
                "best_acc_round": int(cdf.loc[best_idx, "round"]),
                "mean_acc": float(cdf["test_acc_mean"].mean()),
                "std_acc": float(cdf["test_acc_mean"].std(ddof=0) if len(cdf) > 1 else 0.0),
                "n_rounds": int(cdf["round"].nunique()),
                "avg_size": float(cdf["size"].mean()) if "size" in cdf.columns else np.nan,
            })
        per_cluster_summary = clusters_df.groupby("cluster_id").apply(_summ).reset_index()
        per_cluster_summary.to_csv(PLOTS_DIR / "per_cluster_test_accuracy_summary.csv", index=False)
except Exception as e:
    print(f"Export error (per-cluster summary): {e}")

try:
    overview_df.to_csv(PLOTS_DIR / "overview_series.csv", index=False)
except Exception as e:
    print(f"Export error (overview_series): {e}")

# Main CFL figure (2x2) - matches reference implementation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CIFAR-10 CFL Metrics", fontsize=16, fontweight='bold')

# Top-left: Per-cluster Test Accuracy with overall client mean overlay
ax1 = axes[0, 0]
if not clusters_df.empty and {"round","cluster_id","test_acc_mean"}.issubset(clusters_df.columns):
    for cid, cdf in clusters_df.sort_values(["round"]).groupby("cluster_id"):
        ax1.plot(cdf["round"], cdf["test_acc_mean"], linewidth=1.8, label=f"Cluster {cid}")
        if "test_acc_std" in cdf.columns:
            lo = np.clip(cdf["test_acc_mean"] - cdf["test_acc_std"], 0.0, 1.0)
            hi = np.clip(cdf["test_acc_mean"] + cdf["test_acc_std"], 0.0, 1.0)
            ax1.fill_between(cdf["round"], lo, hi, alpha=0.12)

# Overall mean test accuracy across all clients (overlay)
overall_mean = None
if not clients_df.empty and {"round","test_acc"}.issubset(clients_df.columns):
    overall = clients_df.groupby("round")["test_acc"].mean().reset_index()
    overall_mean = overall
elif not clusters_df.empty and {"round","size","test_acc_mean"}.issubset(clusters_df.columns):
    tmp = clusters_df[["round","size","test_acc_mean"]].copy()
    grouped = tmp.groupby("round").apply(
        lambda df: pd.Series({
            "overall": (df["size"] * df["test_acc_mean"]).sum() / max(df["size"].sum(), 1)
        })
    ).reset_index()
    overall_mean = grouped.rename(columns={"overall": "test_acc"})

if overall_mean is not None:
    ax1.plot(overall_mean["round"], overall_mean["test_acc"], color="black", linewidth=2.5, label="Overall mean (clients)")

ax1.set_title("Cluster Test Accuracy vs Round")
ax1.set_xlabel("Round")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right", ncol=2, fontsize=8)

# Top-right: Per-cluster Test Loss
ax2 = axes[0, 1]
if not clusters_df.empty and {"round","cluster_id","test_loss_mean"}.issubset(clusters_df.columns):
    for cid, cdf in clusters_df.sort_values(["round"]).groupby("cluster_id"):
        ax2.plot(cdf["round"], cdf["test_loss_mean"], linewidth=1.8, label=f"Cluster {cid}")
        if "test_loss_std" in cdf.columns:
            lo = cdf["test_loss_mean"] - cdf["test_loss_std"]
            hi = cdf["test_loss_mean"] + cdf["test_loss_std"]
            ax2.fill_between(cdf["round"], lo, hi, alpha=0.12)
ax2.set_title("Cluster Test Loss vs Round")
ax2.set_xlabel("Round")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", ncol=2, fontsize=8)

# Bottom-left: Total Communication (MB) per Round from cluster sums
ax3 = axes[1, 0]
comm_cols = [
    "download_MB_eval_sum","upload_MB_eval_sum",
    "download_MB_fit_sum","upload_MB_fit_sum",
]
if not clusters_df.empty and set(comm_cols).issubset(clusters_df.columns):
    per_round = clusters_df.groupby("round")[comm_cols].sum().reset_index()
    per_round["total_comm_MB"] = per_round[comm_cols].sum(axis=1)
    ax3.plot(per_round["round"], per_round["total_comm_MB"], 'g-', linewidth=2, label="Total Communication")
ax3.set_title("Total Communication Cost per Round")
ax3.set_xlabel("Round")
ax3.set_ylabel("Communication (MB)")
ax3.grid(True, alpha=0.3)
ax3.legend()

# Bottom-right: Wall-clock time per round (+ #clusters on secondary axis)
ax4 = axes[1, 1]
plotted_time = False
if not system_df.empty and {"round","round_wall_time_sec"}.issubset(system_df.columns):
    ax4.plot(system_df["round"], system_df["round_wall_time_sec"], 'purple', linewidth=2, label="Wall time (s)")
    plotted_time = True
if (not plotted_time) and (not clusters_df.empty) and ("comp_time_fit_sec_sum" in clusters_df.columns):
    per_round_time = clusters_df.groupby("round")["comp_time_fit_sec_sum"].sum().reset_index()
    ax4.plot(per_round_time["round"], per_round_time["comp_time_fit_sec_sum"], 'purple', linewidth=2, label="Fit time sum (s)")

# Secondary axis: number of clusters
if not system_df.empty and {"round","num_clusters"}.issubset(system_df.columns):
    ax4b = ax4.twinx()
    ax4b.plot(system_df["round"], system_df["num_clusters"], color="tab:orange", linestyle="--", label="#Clusters")
    ax4b.set_ylabel("#Clusters")
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)
else:
    ax4.legend(loc="upper left", fontsize=8)

ax4.set_title("Computation Time per Round")
ax4.set_xlabel("Round")
ax4.set_ylabel("Time (seconds)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = PLOTS_DIR / "fl_metrics.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Main CFL plot saved to: {output_path}")

# Per-cluster small multiples: Test Accuracy
if not clusters_df.empty and {"round","cluster_id","test_acc_mean"}.issubset(clusters_df.columns):
    cluster_ids = sorted(clusters_df["cluster_id"].unique())
    n = len(cluster_ids)
    ncols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig_pa, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=True)
    axs = (np.array(axs).reshape(nrows, ncols)) if nrows*ncols > 1 else np.array([[axs]])
    for i, cid in enumerate(cluster_ids):
        r, c = divmod(i, ncols)
        ax = axs[r, c]
        cdf = clusters_df[clusters_df["cluster_id"] == cid].sort_values(["round"])
        ax.plot(cdf["round"], cdf["test_acc_mean"], label="Test Acc", color="tab:blue")
        if "test_acc_std" in cdf.columns:
            lo = np.clip(cdf["test_acc_mean"] - cdf["test_acc_std"], 0.0, 1.0)
            hi = np.clip(cdf["test_acc_mean"] + cdf["test_acc_std"], 0.0, 1.0)
            ax.fill_between(cdf["round"], lo, hi, alpha=0.15, color="tab:blue")
        if "train_acc_mean" in cdf.columns:
            ax.plot(cdf["round"], cdf["train_acc_mean"], linestyle="--", color="tab:green", label="Train Acc")
        ax.set_title(f"Cluster {cid}")
        ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3)
    # remove empty panels
    for j in range(i+1, nrows*ncols):
        r, c = divmod(j, ncols)
        fig_pa.delaxes(axs[r, c])
    handles, labels = (axs[0,0].get_legend_handles_labels() if n > 0 else ([], []))
    if handles:
        fig_pa.legend(handles, labels, loc="lower center", ncol=2)
    fig_pa.suptitle("Per-Cluster Test Accuracy (mean±std)", y=1.02, fontsize=14, fontweight='bold')
    fig_pa.tight_layout()
    fig_pa.savefig(PLOTS_DIR / "per_cluster_accuracy.png", dpi=150, bbox_inches='tight')
    print(f"Per-cluster accuracy multiples saved to: {PLOTS_DIR / 'per_cluster_accuracy.png'}")

# Per-cluster small multiples: Test Loss
if not clusters_df.empty and {"round","cluster_id","test_loss_mean"}.issubset(clusters_df.columns):
    cluster_ids = sorted(clusters_df["cluster_id"].unique())
    n = len(cluster_ids)
    ncols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig_pl, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=True)
    axs = (np.array(axs).reshape(nrows, ncols)) if nrows*ncols > 1 else np.array([[axs]])
    for i, cid in enumerate(cluster_ids):
        r, c = divmod(i, ncols)
        ax = axs[r, c]
        cdf = clusters_df[clusters_df["cluster_id"] == cid].sort_values(["round"])
        ax.plot(cdf["round"], cdf["test_loss_mean"], label="Test Loss", color="tab:orange")
        if "test_loss_std" in cdf.columns:
            lo = cdf["test_loss_mean"] - cdf["test_loss_std"]
            hi = cdf["test_loss_mean"] + cdf["test_loss_std"]
            ax.fill_between(cdf["round"], lo, hi, alpha=0.15, color="tab:orange")
        if "train_loss_mean" in cdf.columns:
            ax.plot(cdf["round"], cdf["train_loss_mean"], linestyle="--", color="tab:red", label="Train Loss")
        ax.set_title(f"Cluster {cid}")
        ax.set_xlabel("Round"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    for j in range(i+1, nrows*ncols):
        r, c = divmod(j, ncols)
        fig_pl.delaxes(axs[r, c])
    handles, labels = (axs[0,0].get_legend_handles_labels() if n > 0 else ([], []))
    if handles:
        fig_pl.legend(handles, labels, loc="lower center", ncol=2)
    fig_pl.suptitle("Per-Cluster Test Loss (mean±std)", y=1.02, fontsize=14, fontweight='bold')
    fig_pl.tight_layout()
    fig_pl.savefig(PLOTS_DIR / "per_cluster_loss.png", dpi=150, bbox_inches='tight')
    print(f"Per-cluster loss multiples saved to: {PLOTS_DIR / 'per_cluster_loss.png'}")

# Cluster sizes over time (auto-derive if missing)
sizes_df = clusters_over_time_df
if (sizes_df is None) or sizes_df.empty:
    if not clients_df.empty and {"round","cluster_id","cid"}.issubset(clients_df.columns):
        sizes_df = (
            clients_df.groupby(["round","cluster_id"])["cid"].nunique()
            .reset_index().rename(columns={"cid": "size"})
        )
    else:
        sizes_df = pd.DataFrame()

if not sizes_df.empty and {"round","cluster_id","size"}.issubset(sizes_df.columns):
    fig_sz, ax_sz = plt.subplots(1, 1, figsize=(10, 4))
    for cid, g in sizes_df.sort_values(["round"]).groupby("cluster_id"):
        ax_sz.plot(g["round"], g["size"], label=f"Cluster {cid}")
    ax_sz.set_title("Cluster Sizes over Time")
    ax_sz.set_xlabel("Round"); ax_sz.set_ylabel("Cluster Size (#clients)")
    ax_sz.grid(True, alpha=0.3); ax_sz.legend(ncol=4, fontsize=8)
    fig_sz.tight_layout()
    fig_sz.savefig(PLOTS_DIR / "cluster_sizes.png", dpi=150, bbox_inches='tight')
    print(f"Cluster sizes plot saved to: {PLOTS_DIR / 'cluster_sizes.png'}")

# CFL Overview — metrics (2x2 subplot)
fig_overview, axes_ov = plt.subplots(2, 2, figsize=(12, 8))
fig_overview.suptitle("CFL Overview — metrics", fontsize=16, fontweight='bold')

# Top-left: Macro Avg Cluster Test Accuracy
ax_ov1 = axes_ov[0, 0]
if "macro_cluster_test_acc" in overview_df.columns:
    ax_ov1.plot(overview_df["round"], overview_df["macro_cluster_test_acc"], 
                color="blue", linewidth=2, label="Macro Avg Cluster Test Acc")
ax_ov1.set_title("Macro Avg Cluster Test Accuracy")
ax_ov1.set_xlabel("Round")
ax_ov1.set_ylabel("Test Accuracy")
ax_ov1.set_ylim(0, 1.0)
ax_ov1.grid(True, alpha=0.3)
ax_ov1.legend()

# Top-right: Macro Avg Cluster Test Loss
ax_ov2 = axes_ov[0, 1]
# Compute macro cluster test loss from clusters_df if available
if not clusters_df.empty and {"round","test_loss_mean"}.issubset(clusters_df.columns):
    macro_loss = (
        clusters_df.groupby("round")["test_loss_mean"].mean().reset_index()
        .rename(columns={"test_loss_mean": "macro_cluster_test_loss"})
    )
    ax_ov2.plot(macro_loss["round"], macro_loss["macro_cluster_test_loss"], 
                color="red", linewidth=2, label="Macro Avg Cluster Test Loss")
elif "avg_client_test_loss" in overview_df.columns:
    # Fallback to client average if cluster data unavailable
    ax_ov2.plot(overview_df["round"], overview_df["avg_client_test_loss"], 
                color="red", linewidth=2, label="Avg Client Test Loss")
ax_ov2.set_title("Macro Avg Cluster Test Loss")
ax_ov2.set_xlabel("Round")
ax_ov2.set_ylabel("Test Loss")
ax_ov2.grid(True, alpha=0.3)
ax_ov2.legend()

# Bottom-left: Average Client Test Accuracy
ax_ov3 = axes_ov[1, 0]
if "avg_client_test_acc" in overview_df.columns:
    ax_ov3.plot(overview_df["round"], overview_df["avg_client_test_acc"], 
                color="green", linewidth=2, label="Avg Client Test Acc")
ax_ov3.set_title("Average Client Test Accuracy")
ax_ov3.set_xlabel("Round")
ax_ov3.set_ylabel("Test Accuracy")
ax_ov3.set_ylim(0, 1.0)
ax_ov3.grid(True, alpha=0.3)
ax_ov3.legend()

# Bottom-right: Average Client Test Loss
ax_ov4 = axes_ov[1, 1]
if "avg_client_test_loss" in overview_df.columns:
    ax_ov4.plot(overview_df["round"], overview_df["avg_client_test_loss"], 
                color="orange", linewidth=2, label="Avg Client Test Loss")
ax_ov4.set_title("Average Client Test Loss")
ax_ov4.set_xlabel("Round")
ax_ov4.set_ylabel("Test Loss")
ax_ov4.grid(True, alpha=0.3)
ax_ov4.legend()

plt.tight_layout()
fig_overview.savefig(PLOTS_DIR / "cfl_overview_metrics.png", dpi=150, bbox_inches='tight')
print(f"CFL Overview metrics plot saved to: {PLOTS_DIR / 'cfl_overview_metrics.png'}")

print("CFL cluster analysis complete! Check the plots/ directory for outputs.")
