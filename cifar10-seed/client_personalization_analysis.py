#!/usr/bin/env python3
"""
Client Personalization Analysis for CIFAR-10 Federated Learning

This script analyzes client-level personalization across federated learning methods:
- Client performance distributions and fairness
- Personalization effectiveness metrics  
- Client trajectory analysis
- Statistical tables for documentation

Generates publication-ready plots and comprehensive analysis tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

class ClientPersonalizationAnalyzer:
    def __init__(self, data_dir=".", output_dir="personalization_analysis"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Method configuration
        self.methods = {
            'cfl': {'name': 'CFL', 'color': '#9467bd', 'has_clients': True},
            'fedprox': {'name': 'FedProx', 'color': '#ff7f0e', 'has_clients': False},
            'hierfl': {'name': 'HierFL', 'color': '#d62728', 'has_clients': False},
            'pfedme': {'name': 'pFedMe', 'color': '#2ca02c', 'has_clients': False},
            'scaffold': {'name': 'SCAFFOLD', 'color': '#1f77b4', 'has_clients': False},
            'fedge': {'name': 'Fedge', 'color': '#8c564b', 'has_clients': True}
        }
        
        self.seeds = [0, 1]
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_distributed_metrics(self, method, seed):
        """Load distributed metrics (client-level aggregated data)."""
        filename = f"distributed_metrics_{method}_seed{seed}.csv"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            return pd.read_csv(filepath)
        return None
    
    def load_client_metrics(self, method, seed):
        """Load individual client metrics (detailed per-client data)."""
        filename = f"clients_metrics_{method}_seed{seed}.csv"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            return pd.read_csv(filepath)
        return None
    
    def extract_client_accuracies_from_distributed(self, df):
        """Extract individual client accuracies from distributed metrics."""
        client_cols = [col for col in df.columns if col.startswith('client_') and col.endswith('_accuracy')]
        if not client_cols:
            return None
        
        # Get final round client accuracies
        final_round = df.iloc[-1]
        client_accuracies = []
        
        for col in client_cols:
            if pd.notna(final_round[col]):
                client_accuracies.append(final_round[col])
        
        return np.array(client_accuracies)
    
    def calculate_fairness_metrics(self, client_accuracies):
        """Calculate various fairness and personalization metrics."""
        if len(client_accuracies) == 0:
            return {}
        
        mean_acc = np.mean(client_accuracies)
        std_acc = np.std(client_accuracies)
        min_acc = np.min(client_accuracies)
        max_acc = np.max(client_accuracies)
        
        # Coefficient of Variation (lower is more fair)
        cv = std_acc / mean_acc if mean_acc > 0 else np.inf
        
        # Performance Gap (lower is more fair)
        perf_gap = max_acc - min_acc
        
        # Gini Coefficient (0 = perfect equality, 1 = perfect inequality)
        sorted_acc = np.sort(client_accuracies)
        n = len(sorted_acc)
        cumsum = np.cumsum(sorted_acc)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Percentiles
        p25, p50, p75 = np.percentile(client_accuracies, [25, 50, 75])
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc,
            'p25_accuracy': p25,
            'p50_accuracy': p50,
            'p75_accuracy': p75,
            'coefficient_variation': cv,
            'performance_gap': perf_gap,
            'gini_coefficient': gini,
            'num_clients': len(client_accuracies)
        }
    
    def analyze_client_fairness(self):
        """Analyze client fairness across all methods and seeds."""
        results = []
        
        for method_key, method_info in self.methods.items():
            method_name = method_info['name']
            
            for seed in self.seeds:
                # Try distributed metrics first (has individual client data)
                df_dist = self.load_distributed_metrics(method_key, seed)
                client_accuracies = None
                
                if df_dist is not None:
                    client_accuracies = self.extract_client_accuracies_from_distributed(df_dist)
                
                # If no client data in distributed, try client metrics file
                if client_accuracies is None:
                    df_clients = self.load_client_metrics(method_key, seed)
                    if df_clients is not None:
                        # Get final round accuracies per client
                        final_round = df_clients['round'].max()
                        final_clients = df_clients[df_clients['round'] == final_round]
                        if len(final_clients) > 0:
                            client_accuracies = final_clients['test_accuracy'].values
                
                if client_accuracies is not None and len(client_accuracies) > 0:
                    fairness_metrics = self.calculate_fairness_metrics(client_accuracies)
                    fairness_metrics.update({
                        'method': method_name,
                        'seed': seed
                    })
                    results.append(fairness_metrics)
        
        # Save detailed results
        if results:
            df_results = pd.DataFrame(results)
            output_path = self.output_dir / "client_fairness_detailed.csv"
            df_results.to_csv(output_path, index=False)
            print(f"✅ Client fairness analysis saved to {output_path}")
            
            # Create aggregated summary across seeds
            summary_results = []
            for method_name in df_results['method'].unique():
                method_data = df_results[df_results['method'] == method_name]
                
                if len(method_data) > 0:
                    summary = {
                        'method': method_name,
                        'mean_accuracy_avg': method_data['mean_accuracy'].mean(),
                        'mean_accuracy_std': method_data['mean_accuracy'].std(),
                        'fairness_cv_avg': method_data['coefficient_variation'].mean(),
                        'fairness_cv_std': method_data['coefficient_variation'].std(),
                        'performance_gap_avg': method_data['performance_gap'].mean(),
                        'performance_gap_std': method_data['performance_gap'].std(),
                        'gini_coefficient_avg': method_data['gini_coefficient'].mean(),
                        'gini_coefficient_std': method_data['gini_coefficient'].std(),
                        'num_clients': method_data['num_clients'].iloc[0],
                        'seeds_analyzed': len(method_data)
                    }
                    summary_results.append(summary)
            
            if summary_results:
                df_summary = pd.DataFrame(summary_results)
                summary_path = self.output_dir / "client_fairness_summary.csv"
                df_summary.to_csv(summary_path, index=False)
                print(f"✅ Client fairness summary saved to {summary_path}")
                
                return df_results, df_summary
        
        return None, None
    
    def plot_client_performance_distributions(self):
        """Create box plots and violin plots of client performance distributions."""
        # Collect all client data
        all_client_data = []
        
        for method_key, method_info in self.methods.items():
            method_name = method_info['name']
            method_color = method_info['color']
            
            for seed in self.seeds:
                df_dist = self.load_distributed_metrics(method_key, seed)
                client_accuracies = None
                
                if df_dist is not None:
                    client_accuracies = self.extract_client_accuracies_from_distributed(df_dist)
                
                if client_accuracies is None:
                    df_clients = self.load_client_metrics(method_key, seed)
                    if df_clients is not None:
                        final_round = df_clients['round'].max()
                        final_clients = df_clients[df_clients['round'] == final_round]
                        if len(final_clients) > 0:
                            client_accuracies = final_clients['test_accuracy'].values
                
                if client_accuracies is not None and len(client_accuracies) > 0:
                    for acc in client_accuracies:
                        all_client_data.append({
                            'method': method_name,
                            'seed': seed,
                            'accuracy': acc,
                            'color': method_color
                        })
        
        if not all_client_data:
            print("⚠️ No client-level data found for distribution plots")
            return
        
        df_plot = pd.DataFrame(all_client_data)
        
        # Create box plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Box plot
        methods_order = df_plot['method'].unique()
        colors = [self.methods[k]['color'] for k, v in self.methods.items() if v['name'] in methods_order]
        
        box_plot = ax1.boxplot([df_plot[df_plot['method'] == method]['accuracy'].values 
                               for method in methods_order],
                              labels=methods_order,
                              patch_artist=True,
                              showmeans=True,
                              meanline=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Client Performance Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_xlabel('Method', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plot
        violin_parts = ax2.violinplot([df_plot[df_plot['method'] == method]['accuracy'].values 
                                      for method in methods_order],
                                     positions=range(1, len(methods_order) + 1),
                                     showmeans=True,
                                     showmedians=True)
        
        # Color the violins
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax2.set_title('Client Performance Density (Violin Plot)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Test Accuracy', fontsize=12)
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_xticks(range(1, len(methods_order) + 1))
        ax2.set_xticklabels(methods_order, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'client_performance_distributions.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Client performance distribution plots saved")
    
    def plot_fairness_comparison(self):
        """Create bar plots comparing fairness metrics across methods."""
        # Load fairness summary
        summary_path = self.output_dir / "client_fairness_summary.csv"
        if not summary_path.exists():
            print("⚠️ Run analyze_client_fairness() first")
            return
        
        df_summary = pd.read_csv(summary_path)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = df_summary['method']
        colors = [self.methods[k]['color'] for k, v in self.methods.items() if v['name'] in methods.values]
        
        # 1. Performance Gap
        bars1 = ax1.bar(methods, df_summary['performance_gap_avg'], 
                       yerr=df_summary['performance_gap_std'],
                       color=colors, alpha=0.8, capsize=5)
        ax1.set_title('Performance Gap (Max - Min Client Accuracy)', fontweight='bold')
        ax1.set_ylabel('Accuracy Gap')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars1, df_summary['performance_gap_avg'], df_summary['performance_gap_std']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Coefficient of Variation
        bars2 = ax2.bar(methods, df_summary['fairness_cv_avg'],
                       yerr=df_summary['fairness_cv_std'],
                       color=colors, alpha=0.8, capsize=5)
        ax2.set_title('Coefficient of Variation (Std/Mean)', fontweight='bold')
        ax2.set_ylabel('CV (lower = more fair)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Gini Coefficient
        bars3 = ax3.bar(methods, df_summary['gini_coefficient_avg'],
                       yerr=df_summary['gini_coefficient_std'],
                       color=colors, alpha=0.8, capsize=5)
        ax3.set_title('Gini Coefficient (0=Equal, 1=Unequal)', fontweight='bold')
        ax3.set_ylabel('Gini Coefficient')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Mean Client Accuracy
        bars4 = ax4.bar(methods, df_summary['mean_accuracy_avg'],
                       yerr=df_summary['mean_accuracy_std'],
                       color=colors, alpha=0.8, capsize=5)
        ax4.set_title('Average Client Performance', fontweight='bold')
        ax4.set_ylabel('Mean Client Accuracy')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fairness_metrics_comparison.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Fairness comparison plots saved")
    
    def plot_client_trajectories(self, method='scaffold', seed=0, max_clients=10):
        """Plot individual client learning trajectories over rounds."""
        df_dist = self.load_distributed_metrics(method, seed)
        
        if df_dist is None:
            print(f"⚠️ No distributed metrics found for {method} seed {seed}")
            return
        
        # Extract client accuracy columns
        client_cols = [col for col in df_dist.columns if col.startswith('client_') and col.endswith('_accuracy')]
        
        if not client_cols:
            print(f"⚠️ No individual client data found for {method}")
            return
        
        # Limit number of clients for readability
        client_cols = client_cols[:max_clients]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        rounds = df_dist['round']
        colors = plt.cm.tab10(np.linspace(0, 1, len(client_cols)))
        
        for i, col in enumerate(client_cols):
            client_id = col.replace('client_', '').replace('_accuracy', '')
            client_acc = df_dist[col]
            
            # Only plot if we have valid data
            valid_mask = pd.notna(client_acc)
            if valid_mask.sum() > 0:
                ax.plot(rounds[valid_mask], client_acc[valid_mask], 
                       color=colors[i], linewidth=2, alpha=0.8, 
                       label=f'Client {client_id}')
        
        # Add average line
        avg_acc = df_dist['avg_accuracy']
        ax.plot(rounds, avg_acc, color='black', linewidth=3, 
               linestyle='--', label='Average', alpha=0.9)
        
        ax.set_title(f'Client Learning Trajectories - {self.methods[method]["name"]} (Seed {seed})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'client_trajectories_{method}_seed{seed}.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Client trajectories plot saved for {method} seed {seed}")
    
    def create_personalization_summary_table(self):
        """Create a comprehensive summary table for documentation."""
        summary_path = self.output_dir / "client_fairness_summary.csv"
        if not summary_path.exists():
            print("⚠️ Run analyze_client_fairness() first")
            return
        
        df_summary = pd.read_csv(summary_path)
        
        # Create formatted table for documentation
        table_data = []
        for _, row in df_summary.iterrows():
            table_data.append({
                'Method': row['method'],
                'Avg Client Accuracy': f"{row['mean_accuracy_avg']:.3f} ± {row['mean_accuracy_std']:.3f}",
                'Performance Gap': f"{row['performance_gap_avg']:.3f} ± {row['performance_gap_std']:.3f}",
                'Coefficient of Variation': f"{row['fairness_cv_avg']:.3f} ± {row['fairness_cv_std']:.3f}",
                'Gini Coefficient': f"{row['gini_coefficient_avg']:.3f} ± {row['gini_coefficient_std']:.3f}",
                'Num Clients': int(row['num_clients']),
                'Seeds': int(row['seeds_analyzed'])
            })
        
        df_table = pd.DataFrame(table_data)
        
        # Save as CSV
        table_path = self.output_dir / "personalization_summary_table.csv"
        df_table.to_csv(table_path, index=False)
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df_table.values,
                        colLabels=df_table.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(df_table) + 1):
            for j in range(len(df_table.columns)):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.title('Client Personalization & Fairness Analysis Summary\n(Mean ± Std across seeds)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(self.output_dir / 'personalization_summary_table.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Personalization summary table saved")
        return df_table
    
    def generate_personalization_report(self):
        """Generate a comprehensive personalization analysis report."""
        report_path = self.output_dir / "PERSONALIZATION_REPORT.md"
        
        # Load summary data
        summary_path = self.output_dir / "client_fairness_summary.csv"
        if not summary_path.exists():
            print("⚠️ Run analyze_client_fairness() first")
            return
        
        df_summary = pd.read_csv(summary_path)
        
        # Generate report content
        report_content = f"""# Client Personalization Analysis Report

## Overview

This report analyzes client-level personalization and fairness across {len(df_summary)} federated learning methods on CIFAR-10. The analysis focuses on how well each method handles client heterogeneity and provides fair performance across all participating clients.

## Key Metrics Analyzed

1. **Performance Gap**: Difference between best and worst performing clients (lower = more fair)
2. **Coefficient of Variation**: Standard deviation divided by mean (lower = more consistent)
3. **Gini Coefficient**: Inequality measure from economics (0 = perfect equality, 1 = perfect inequality)
4. **Average Client Performance**: Mean accuracy across all clients

## Results Summary

### Client Fairness Ranking (by Performance Gap - lower is better):
"""
        
        # Sort by performance gap (lower is more fair)
        df_sorted = df_summary.sort_values('performance_gap_avg')
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            report_content += f"{i}. **{row['method']}**: {row['performance_gap_avg']:.3f} ± {row['performance_gap_std']:.3f}\n"
        
        report_content += f"""

### Detailed Analysis

| Method | Avg Client Accuracy | Performance Gap | Coefficient of Variation | Gini Coefficient |
|--------|-------------------|-----------------|-------------------------|------------------|
"""
        
        for _, row in df_summary.iterrows():
            report_content += f"| **{row['method']}** | {row['mean_accuracy_avg']:.3f} ± {row['mean_accuracy_std']:.3f} | {row['performance_gap_avg']:.3f} ± {row['performance_gap_std']:.3f} | {row['fairness_cv_avg']:.3f} ± {row['fairness_cv_std']:.3f} | {row['gini_coefficient_avg']:.3f} ± {row['gini_coefficient_std']:.3f} |\n"
        
        # Find best and worst methods
        best_fairness = df_sorted.iloc[0]
        worst_fairness = df_sorted.iloc[-1]
        best_performance = df_summary.loc[df_summary['mean_accuracy_avg'].idxmax()]
        
        report_content += f"""

## Key Findings

### Fairness Champion: **{best_fairness['method']}**
- Lowest performance gap: {best_fairness['performance_gap_avg']:.3f} ± {best_fairness['performance_gap_std']:.3f}
- Most equitable client treatment across {int(best_fairness['num_clients'])} clients

### Performance Leader: **{best_performance['method']}**  
- Highest average client accuracy: {best_performance['mean_accuracy_avg']:.3f} ± {best_performance['mean_accuracy_std']:.3f}
- Demonstrates strong personalization effectiveness

### Personalization Insights:
1. **{best_fairness['method']}** provides the most equitable performance distribution
2. **{best_performance['method']}** achieves the highest client-level performance
3. Performance gaps range from {df_summary['performance_gap_avg'].min():.3f} to {df_summary['performance_gap_avg'].max():.3f}
4. All methods analyzed across {df_summary['seeds_analyzed'].iloc[0]} random seeds for robustness

## Recommendations for Documentation

### For Papers/Reports:
- Emphasize fairness metrics alongside accuracy results
- Report both individual client performance and aggregate statistics
- Use performance gap as primary fairness indicator
- Include client trajectory plots for visual impact

### Best Practices:
- Always report personalization metrics with confidence intervals
- Consider both fairness and performance when evaluating methods
- Analyze client heterogeneity in context of data distribution
- Document number of clients and seeds for reproducibility

## Generated Files

All analysis results saved in `{self.output_dir}/`:
- `client_fairness_detailed.csv` - Per-seed detailed metrics
- `client_fairness_summary.csv` - Aggregated summary statistics  
- `personalization_summary_table.csv` - Formatted table for documentation
- `client_performance_distributions.png` - Box/violin plots
- `fairness_metrics_comparison.png` - Fairness comparison charts
- `client_trajectories_*.png` - Individual client learning curves
- `personalization_summary_table.png` - Visual summary table

---

*Analysis completed on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"✅ Personalization analysis report saved to {report_path}")
        
    def run_complete_analysis(self):
        """Run the complete personalization analysis pipeline."""
        print("🚀 Starting comprehensive client personalization analysis...")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Output directory: {self.output_dir}")
        print()
        
        # 1. Analyze client fairness
        print("1️⃣ Analyzing client fairness metrics...")
        detailed, summary = self.analyze_client_fairness()
        
        if summary is None:
            print("❌ No client-level data found. Analysis cannot proceed.")
            return
        
        # 2. Create distribution plots
        print("\n2️⃣ Creating client performance distribution plots...")
        self.plot_client_performance_distributions()
        
        # 3. Create fairness comparison plots
        print("\n3️⃣ Creating fairness comparison plots...")
        self.plot_fairness_comparison()
        
        # 4. Create client trajectories (for methods with individual client data)
        print("\n4️⃣ Creating client trajectory plots...")
        for method_key in ['scaffold', 'cfl', 'fedge']:  # Methods likely to have client data
            if method_key in self.methods:
                self.plot_client_trajectories(method_key, seed=0)
        
        # 5. Create summary table
        print("\n5️⃣ Creating personalization summary table...")
        table = self.create_personalization_summary_table()
        
        # 6. Generate comprehensive report
        print("\n6️⃣ Generating personalization analysis report...")
        self.generate_personalization_report()
        
        print("\n✨ Personalization analysis complete!")
        print(f"📋 Check '{self.output_dir}/' for all generated files and reports.")
        
        return {
            'detailed_metrics': detailed,
            'summary_metrics': summary,
            'summary_table': table
        }

def main():
    """Main execution function."""
    analyzer = ClientPersonalizationAnalyzer(
        data_dir=".",
        output_dir="personalization_analysis"
    )
    
    results = analyzer.run_complete_analysis()
    
    print("\n🎯 Key Recommendations for Documenting Personalization:")
    print("   • Use Performance Gap as primary fairness metric")
    print("   • Include client distribution plots in papers")
    print("   • Report Gini coefficient for economic interpretation")
    print("   • Show individual client trajectories for visual impact")
    print("   • Always include confidence intervals across seeds")

if __name__ == "__main__":
    main()
