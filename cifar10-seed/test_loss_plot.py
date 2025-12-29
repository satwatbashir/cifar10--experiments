#!/usr/bin/env python3
"""Test the loss convergence plot."""

from plot_results import plot_loss_convergence_curves

print("🔥 Testing loss convergence plot...")
try:
    plot_loss_convergence_curves()
    print("✅ Loss convergence plot generated successfully!")
    print("📁 Saved as: metrics/summary/loss_convergence_curves.png")
except Exception as e:
    print(f"❌ Error generating loss plot: {e}")
    import traceback
    traceback.print_exc()
