#!/usr/bin/env python3
"""
Plotting script for CIFAR-10 federated learning seed aggregation results.
Generates publication-ready plots from the analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for publication-quality plots with white background
plt.style.use('default')  # Use default style for white background
sns.set_palette("husl")

def plot_final_performance():
    """Plot final performance comparison across methods."""
    # Load data
    df = pd.read_csv('metrics/summary/seed_means.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy plot
    methods = df['method']
    acc_means = df['final_acc_mean']
    acc_stds = df['final_acc_std']
    
    bars1 = ax1.bar(methods, acc_means, yerr=acc_stds, capsize=5, alpha=0.8)
    ax1.set_ylabel('Final Test Accuracy')
    ax1.set_title('Final Test Accuracy by Method')
    ax1.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, acc_means, acc_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Loss plot
    loss_means = df['final_loss_mean']
    loss_stds = df['final_loss_std']
    
    bars2 = ax2.bar(methods, loss_means, yerr=loss_stds, capsize=5, alpha=0.8, color='orange')
    ax2.set_ylabel('Final Test Loss')
    ax2.set_title('Final Test Loss by Method')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars2, loss_means, loss_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('metrics/summary/final_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence_curves():
    """Plot convergence curves without confidence bands."""
    curves_dir = Path('metrics/summary/curves')
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Define colors and method mapping to match the reference plot
    method_colors = {
        'SCAFFOLD': '#1f77b4',  # Blue
        'FEDPROX': '#ff7f0e',   # Orange  
        'PFEDME': '#2ca02c',    # Green
        'HIERFL': '#d62728',    # Red
        'CFL': '#9467bd',       # Purple
        'FEDGE-V1': '#8c564b'   # Brown (if present)
    }
    
    # Method name mapping for proper ordering and display
    method_display_names = {
        'scaffold': 'Scaffold',
        'fedprox': 'FedProx', 
        'pfedme': 'pFedMe',
        'hierfl': 'HierFL',
        'cfl': 'CFL',
        'fedge': 'Fedge-v1'
    }
    
    # Define the desired legend order (Fedge-v1 at the end)
    method_order = ['scaffold', 'fedprox', 'pfedme', 'hierfl', 'cfl', 'fedge']
    
    # Plot in the specified order
    for method_key in method_order:
        curve_file = curves_dir / f"{method_key}_mean_curve.csv"
        if curve_file.exists():
            method_name = method_display_names.get(method_key, method_key.upper())
            df = pd.read_csv(curve_file)
            
            rounds = df['round']
            mean_acc = df['mean_accuracy']
            
            # Get color for this method
            color = method_colors.get(method_name.upper(), '#000000')
            
            # Plot mean line only (no confidence bands/shadows)
            ax.plot(rounds, mean_acc, label=method_name, color=color, linewidth=2)
    
    ax.set_xlabel('Global Rounds', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Rounds', fontsize=14, fontweight='bold')
    
    # Set axis limits and ticks to match reference plot
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.0)
    
    # Set x-axis ticks
    ax.set_xticks(range(0, 151, 10))
    
    # Set y-axis ticks
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Add grid with white background
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
    
    # Position legend outside the plot area on the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=False, shadow=False)
    
    plt.tight_layout()
    plt.savefig('metrics/summary/convergence_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_loss_convergence_curves():
    """Plot loss convergence curves without confidence bands."""
    curves_dir = Path('metrics/summary/curves')

    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Define colors and method mapping to match the reference plot
    method_colors = {
        'SCAFFOLD': '#1f77b4',  # Blue
        'FEDPROX': '#ff7f0e',   # Orange  
        'PFEDME': '#2ca02c',    # Green
        'HIERFL': '#d62728',    # Red
        'CFL': '#9467bd',       # Purple
        'FEDGE-V1': '#8c564b'   # Brown
    }

    # Method name mapping for proper ordering and display
    method_display_names = {
        'scaffold': 'Scaffold',
        'fedprox': 'FedProx', 
        'pfedme': 'pFedMe',
        'hierfl': 'HierFL',
        'cfl': 'CFL',
        'fedge': 'Fedge-v1'
    }
    
    # Define the desired legend order (Fedge-v1 at the end)
    method_order = ['scaffold', 'fedprox', 'pfedme', 'hierfl', 'cfl', 'fedge']
    
    # Plot in the specified order
    for method_key in method_order:
        curve_file = curves_dir / f"{method_key}_mean_curve.csv"
        if curve_file.exists():
            method_name = method_display_names.get(method_key, method_key.upper())
            df = pd.read_csv(curve_file)
            
            rounds = df['round']
            mean_loss = df['mean_loss']  # Use loss instead of accuracy
            
            # Get color for this method
            color = method_colors.get(method_name.upper(), '#000000')
            
            # Plot mean line only (no confidence bands/shadows)
            ax.plot(rounds, mean_loss, label=method_name, color=color, linewidth=2)
    
    ax.set_xlabel('Global Rounds', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Loss vs Rounds', fontsize=14, fontweight='bold')

    # Set axis limits and ticks
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 2.5)  # Adjust for loss range

    # Set x-axis ticks
    ax.set_xticks(range(0, 151, 10))

    # Set y-axis ticks for loss
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    # Add grid with white background
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')

    # Position legend outside the plot area on the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=False, shadow=False)

    plt.tight_layout()
    plt.savefig('metrics/summary/loss_convergence_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_both_convergence_curves():
    """Plot both accuracy and loss convergence curves."""
    print("📈 Generating accuracy convergence curves...")
    plot_convergence_curves()
    
    print("📉 Generating loss convergence curves...")
    plot_loss_convergence_curves()
    
    print("✅ Both convergence plots generated!")
    print("   📁 Accuracy: metrics/summary/convergence_curves.png")
    print("   📁 Loss: metrics/summary/loss_convergence_curves.png")

def plot_efficiency_analysis():
    """Plot efficiency metrics comparison."""
    df = pd.read_csv('metrics/summary/efficiency_metrics.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Communication efficiency
    methods = df['method']
    comm_means = df['avg_communication_MB']
    
    bars1 = ax1.bar(methods, comm_means, alpha=0.8, color='skyblue')
    ax1.set_ylabel('Communication (MB)')
    ax1.set_title('Communication Efficiency')
    ax1.set_ylim(0, max(comm_means) * 1.2)
    
    # Add value labels
    for bar, mean in zip(bars1, comm_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f}', ha='center', va='bottom')
    
    # Computation time (where available)
    comp_data = df.dropna(subset=['avg_comp_time_sec'])
    if len(comp_data) > 0:
        comp_methods = comp_data['method']
        comp_means = comp_data['avg_comp_time_sec']
        comp_stds = comp_data['std_comp_time_sec']
        
        bars2 = ax2.bar(comp_methods, comp_means, yerr=comp_stds, 
                       capsize=5, alpha=0.8, color='lightcoral')
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title('Computation Efficiency')
        
        # Add value labels
        for bar, mean, std in zip(bars2, comp_means, comp_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('metrics/summary/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_client_heterogeneity():
    """Plot client heterogeneity analysis."""
    df = pd.read_csv('metrics/summary/client_heterogeneity.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = df['method']
    
    # Client accuracy means
    acc_means = df['client_acc_mean_across_seeds']
    acc_stds = df['client_acc_mean_std_across_seeds']
    
    bars1 = ax1.bar(methods, acc_means, yerr=acc_stds, capsize=5, alpha=0.8, color='lightgreen')
    ax1.set_ylabel('Client Average Accuracy')
    ax1.set_title('Client Performance Across Methods')
    
    # Client heterogeneity (variance)
    het_means = df['client_heterogeneity_mean']
    het_stds = df['client_heterogeneity_std']
    
    bars2 = ax2.bar(methods, het_means, yerr=het_stds, capsize=5, alpha=0.8, color='salmon')
    ax2.set_ylabel('Client Heterogeneity (Std Dev)')
    ax2.set_title('Client Performance Variance')
    
    # Add value labels
    for bar, mean, std in zip(bars2, het_means, het_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('metrics/summary/client_heterogeneity.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table_plot():
    """Create a visual summary table of key results."""
    df = pd.read_csv('metrics/summary/seed_means.csv')
    
    # Prepare data for table
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['method'],
            f"{row['final_acc_mean']:.3f} ± {row['final_acc_std']:.3f}",
            f"{row['final_loss_mean']:.3f} ± {row['final_loss_std']:.3f}",
            f"[{row['final_acc_ci95_lo']:.3f}, {row['final_acc_ci95_hi']:.3f}]"
        ])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                    colLabels=['Method', 'Final Accuracy', 'Final Loss', '95% CI (Accuracy)'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('CIFAR-10 Federated Learning Results Summary\n(Mean ± Std across 2 seeds)', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('metrics/summary/results_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all plots."""
    print("🎨 Generating publication-ready plots...")
    
    # Create output directory
    Path('metrics/summary').mkdir(parents=True, exist_ok=True)
    
    print("1️⃣ Final performance comparison...")
    plot_final_performance()
    
    print("2️⃣ Accuracy convergence curves...")
    plot_convergence_curves()
    
    print("3️⃣ Loss convergence curves...")
    plot_loss_convergence_curves()
    
    print("4️⃣ Efficiency analysis...")
    plot_efficiency_analysis()
    
    print("5️⃣ Client heterogeneity...")
    plot_client_heterogeneity()
    
    print("6️⃣ Summary table...")
    create_summary_table_plot()
    
    print("✨ All plots generated! Check metrics/summary/ for PNG files.")

if __name__ == "__main__":
    main()
