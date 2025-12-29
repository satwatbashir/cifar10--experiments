#!/usr/bin/env python3
"""
CIFAR-10 Federated Learning Results Table (Template-Aligned)

Produces a CSV with EXACT columns as your template:
['Method','Final_Accuracy_%','Std_Dev_%','CI_95%','Generalization_Gap_%','Final_Loss']

- Final_Accuracy_% comes from centralized FINAL accuracy (last round).
- Std_Dev_% and CI_95% come from distributed FINAL-ROUND client accuracies (if available).
- Generalization_Gap_% = (centralized_final_acc - distributed_final_mean_acc) * 100.
- Final_Loss = centralized FINAL loss (last round).

If distributed data is missing, Std_Dev_% and CI_95% are filled as 'N/A',
and Generalization_Gap_% is 'N/A'.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------- Paths ----------------
METRICS_DIR = "cifar10-metrics"
PLOTS_DIR   = "plots"
OUTPUT_FILE = "cifar10_results_summary_template.csv"

# ---------------- Config ----------------
METHODS_CONFIG = {
    "Scaffold": {
        "centralized_file": "scaffold_centralized_metrics.csv",
        "distributed_file":  "scaffold_clients.csv",
        "distributed_files": None,
        "centralized_acc_col": "central_test_accuracy",
        "centralized_loss_col": "central_test_loss",
        "distributed_acc_col": "train_accuracy_mean",  # Use training accuracy as proxy for client-specific performance
        "distributed_loss_col": "train_loss_mean",     # Use training loss as proxy for client-specific performance
        "round_col": "round",
    },
    "FedProx": {
        "centralized_file": "fedprox_centralized_metrics.csv",
        "distributed_file":  "fedprox_distributed_metrics.csv", 
        "distributed_files": None,
        "centralized_acc_col": "central_test_accuracy",
        "centralized_loss_col": "central_test_loss",
        "distributed_acc_col": ["client_1_accuracy", "client_2_accuracy", "client_3_accuracy", "client_4_accuracy", "client_5_accuracy", "client_6_accuracy", "client_7_accuracy", "client_8_accuracy", "client_9_accuracy", "client_10_accuracy"],
        "distributed_loss_col": ["client_1_loss", "client_2_loss", "client_3_loss", "client_4_loss", "client_5_loss", "client_6_loss", "client_7_loss", "client_8_loss", "client_9_loss", "client_10_loss"],
        "round_col": "round",
    },
    "pFedMe": {
        "centralized_file": "pfedme_centralized_metrics.csv",
        "distributed_file":  "pfedme_clients.csv",
        "distributed_files": None,
        "centralized_acc_col": "central_test_accuracy",
        "centralized_loss_col": "central_test_loss",
        "distributed_acc_col": "test_accuracy",
        "distributed_loss_col": "test_loss",
        "round_col": "round",
    },
    "HierFL": {
        "centralized_file": "hierfl_global_model_metrics.csv",
        "distributed_file":  None,
        "distributed_files": [
            "hierfl_server_0_client_eval_metrics.csv",
            "hierfl_server_1_client_eval_metrics.csv",
            "hierfl_server_2_client_eval_metrics.csv",
        ],
        "centralized_acc_col": "global_test_accuracy_centralized",
        "centralized_loss_col": "global_test_loss_centralized",
        "distributed_acc_col": "client_test_accuracy",
        "distributed_loss_col": "client_test_loss",
        "round_col": "global_round",
    },
    "CFL": {
        "centralized_file": "cfl_clusters_metrics.csv",
        "distributed_file":  "cfl_clients_metrics.csv",
        "distributed_files": None,
        "centralized_acc_col": "test_acc_mean",
        "centralized_loss_col": "test_loss_mean",
        "distributed_acc_col": "test_acc",
        "distributed_loss_col": "test_loss",
        "round_col": "round",
    },
    "Fedge-v1": {
        "centralized_file": "fedge_cloud_cloud_round_metrics.csv",
        "distributed_file":  None,
        "distributed_files": [
            "fedge_server_0_client_eval_metrics.csv",
            "fedge_server_1_client_eval_metrics.csv",
            "fedge_server_2_client_eval_metrics.csv",
        ],
        "centralized_acc_col": "cluster_accuracy_mean",
        "centralized_loss_col": "cluster_loss_mean",
        "distributed_acc_col": "accuracy",
        "distributed_loss_col": "eval_loss",
        "round_col": "global_round",
    },
    # If you want Fedge-50 in the table as well, add a block like this:
    # "Fedge-50": { ... same keys with its file names ... },
}

# ---------------- Helpers ----------------
def _read_csv_safe(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return None

def load_centralized(method, cfg):
    path = os.path.join(METRICS_DIR, cfg["centralized_file"])
    df = _read_csv_safe(path)
    if df is None:
        print(f"[warn] Centralized file missing for {method}: {path}")
        return None
    req = [cfg["round_col"], cfg["centralized_acc_col"], cfg["centralized_loss_col"]]
    miss = [c for c in req if c not in df.columns]
    if miss:
        print(f"[warn] {method} centralized missing {miss}. Have: {list(df.columns)}")
        return None
    df = df.dropna(subset=req).copy()
    if df.empty:
        print(f"[warn] {method} centralized empty after NaN drop.")
        return None
    return df

def load_distributed(method, cfg):
    frames = []
    if cfg.get("distributed_files"):
        for fname in cfg["distributed_files"]:
            path = os.path.join(METRICS_DIR, fname)
            df = _read_csv_safe(path)
            if df is None:
                print(f"[warn] Distributed file missing for {method}: {path}")
                continue
            frames.append(df)
    if cfg.get("distributed_file"):
        path = os.path.join(METRICS_DIR, cfg["distributed_file"])
        df = _read_csv_safe(path)
        if df is None:
            print(f"[warn] Distributed file missing for {method}: {path}")
        else:
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    
    # Handle both single column names and lists of column names
    acc_cols = cfg["distributed_acc_col"] if isinstance(cfg["distributed_acc_col"], list) else [cfg["distributed_acc_col"]]
    loss_cols = cfg["distributed_loss_col"] if isinstance(cfg["distributed_loss_col"], list) else [cfg["distributed_loss_col"]]
    req = [cfg["round_col"]] + acc_cols + loss_cols
    
    miss = [c for c in req if c not in df.columns]
    if miss:
        print(f"[warn] {method} distributed missing {miss}. Have: {list(df.columns)}")
        return None
    df = df.dropna(subset=req).copy()
    if df.empty:
        print(f"[warn] {method} distributed empty after NaN drop.")
        return None
    return df

def ci95(values):
    """Return (mean, std, ci_low, ci_high) in FRACTION units (0–1)."""
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n < 2:
        return mean, std, np.nan, np.nan
    sem = std / np.sqrt(n)
    h = float(stats.t.ppf(0.975, n - 1) * sem)
    return mean, std, mean - h, mean + h

def pct(x, digits=2):
    """Convert fraction to percent (0–1 -> 0–100) and round."""
    if x is None or np.isnan(x):
        return None
    return round(x * 100.0, digits)

def fmt_ci95_pct(lo, hi, digits=2):
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return "N/A"
    return f"[{round(lo*100.0, digits):.{digits}f}, {round(hi*100.0, digits):.{digits}f}]"

# ---------------- Build table ----------------
def build_table():
    rows = []
    for method, cfg in METHODS_CONFIG.items():
        # Centralized final
        cen = load_centralized(method, cfg)
        if cen is None:
            # minimally fill row with N/As
            rows.append({
                "Method": method,
                "Final_Accuracy_%": "N/A",
                "Std_Dev_%": "N/A",
                "CI_95%": "N/A",
                "Generalization_Gap_%": "N/A",
                "Final_Loss": "N/A",
            })
            continue

        cen_sorted = cen.sort_values(cfg["round_col"])
        final_acc_frac  = float(cen_sorted[cfg["centralized_acc_col"]].iloc[-1])  # 0–1
        final_loss      = float(cen_sorted[cfg["centralized_loss_col"]].iloc[-1])

        # Distributed final round stats (optional)
        dist = load_distributed(method, cfg)
        if dist is not None:
            last_round = np.nanmax(dist[cfg["round_col"]].to_numpy())
            fr = dist.loc[dist[cfg["round_col"]] == last_round]
            
            # Handle both single column and multiple columns for accuracy
            acc_cols = cfg["distributed_acc_col"] if isinstance(cfg["distributed_acc_col"], list) else [cfg["distributed_acc_col"]]
            
            # Collect all accuracy values from final round
            acc_vals = []
            for col in acc_cols:
                if col in fr.columns:
                    vals = fr[col].to_numpy(dtype=float)
                    acc_vals.extend(vals[~np.isnan(vals)])  # Remove NaN values
            
            if acc_vals:
                acc_vals = np.array(acc_vals)
                # compute mean/std/CI in fraction units
                mean_f, std_f, lo_f, hi_f = ci95(acc_vals)
                std_pct = pct(std_f, 2) if std_f is not None and not np.isnan(std_f) else None
                ci_str  = fmt_ci95_pct(lo_f, hi_f, 2)
                gengap  = None
                if not np.isnan(mean_f):
                    gengap = pct(final_acc_frac - mean_f, 2)
            else:
                std_pct = None
                ci_str  = "N/A"
                gengap  = None
        else:
            std_pct = None
            ci_str  = "N/A"
            gengap  = None

        row = {
            "Method": method,
            "Final_Accuracy_%": f"{pct(final_acc_frac, 2):.2f}" if final_acc_frac==final_acc_frac else "N/A",
            "Std_Dev_%": f"{std_pct:.2f}" if std_pct is not None else "N/A",
            "CI_95%": ci_str,
            "Generalization_Gap_%": f"{gengap:.2f}" if gengap is not None else "N/A",
            "Final_Loss": f"{final_loss:.4f}" if final_loss==final_loss else "N/A",
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "Method",
        "Final_Accuracy_%",
        "Std_Dev_%",
        "CI_95%",
        "Generalization_Gap_%",
        "Final_Loss",
    ])
    return df

def main():
    if not os.path.isdir(METRICS_DIR):
        raise SystemExit(f"[error] Metrics directory not found: {METRICS_DIR}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = build_table()
    out = os.path.join(PLOTS_DIR, OUTPUT_FILE)
    df.to_csv(out, index=False)
    print(f"[ok] Saved: {out}")

if __name__ == "__main__":
    main()
