"""Generate FL metrics plots with four subplots.

This script creates a single PNG with:
1. Top-left: Server accuracy (from centralized CSV)
2. Top-right: Server loss (from centralized CSV)
3. Bottom-left: Total communication cost
4. Bottom-right: Computation time per round
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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

# Load data
centralized_df = read_csv("centralized_metrics.csv")
distributed_df = read_csv("distributed_metrics.csv")
fit_df = read_csv("fit_metrics.csv")

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Federated Learning Metrics", fontsize=16, fontweight='bold')

# Top-left: Server Accuracy
ax1 = axes[0, 0]
if "central_test_accuracy" in centralized_df.columns:
    ax1.plot(centralized_df["round"], centralized_df["central_test_accuracy"], 
             'b-', linewidth=2, label="Test Accuracy")
if "central_train_accuracy" in centralized_df.columns:
    ax1.plot(centralized_df["round"], centralized_df["central_train_accuracy"], 
             'r-', linewidth=2, label="Train Accuracy")
ax1.set_title("Server Accuracy vs. Round")
ax1.set_xlabel("Round")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Top-right: Server Loss
ax2 = axes[0, 1]
if "central_test_loss" in centralized_df.columns:
    ax2.plot(centralized_df["round"], centralized_df["central_test_loss"], 
             'b-', linewidth=2, label="Test Loss")
if "central_train_loss" in centralized_df.columns:
    ax2.plot(centralized_df["round"], centralized_df["central_train_loss"], 
             'r-', linewidth=2, label="Train Loss")
ax2.set_title("Server Loss vs. Round")
ax2.set_xlabel("Round")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)
ax2.legend()

# Bottom-left: Communication Cost
ax3 = axes[1, 0]
# Try distributed_df first, then fit_df
if "total_upload_MB" in distributed_df.columns and "total_download_MB" in distributed_df.columns:
    total_comm = distributed_df["total_upload_MB"] + distributed_df["total_download_MB"]
    ax3.plot(distributed_df["round"], total_comm, 'g-', linewidth=2, label="Total Communication")
elif "total_upload_MB" in fit_df.columns and "total_download_MB" in fit_df.columns:
    total_comm = fit_df["total_upload_MB"] + fit_df["total_download_MB"]
    ax3.plot(fit_df["round"], total_comm, 'g-', linewidth=2, label="Total Communication")
ax3.set_title("Total Communication Cost per Round")
ax3.set_xlabel("Round")
ax3.set_ylabel("Communication (MB)")
ax3.grid(True, alpha=0.3)
ax3.legend()

# Bottom-right: Computation Time
ax4 = axes[1, 1]
if "total_comp_time_sec" in distributed_df.columns:
    ax4.plot(distributed_df["round"], distributed_df["total_comp_time_sec"], 
             'purple', linewidth=2, label="Computation Time")
elif "total_comp_time_sec" in fit_df.columns:
    ax4.plot(fit_df["round"], fit_df["total_comp_time_sec"], 
             'purple', linewidth=2, label="Computation Time")
ax4.set_title("Computation Time per Round")
ax4.set_xlabel("Round")
ax4.set_ylabel("Time (seconds)")
ax4.grid(True, alpha=0.3)
ax4.legend()

# Save the plot
plt.tight_layout()
output_path = PLOTS_DIR / "fl_metrics.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
plt.show()
