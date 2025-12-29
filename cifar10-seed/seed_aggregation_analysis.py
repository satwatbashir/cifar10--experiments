#!/usr/bin/env python3
"""
Comprehensive Seed Aggregation Analysis for CIFAR-10 Federated Learning Results

This script analyzes results across multiple seeds for 5 federated learning methods:
- FedProx, pFedMe, SCAFFOLD, CFL, HierFL (FEDGE)

Generates:
1. Summary statistics (mean ± std ± CI) for final accuracy/loss
2. Convergence analysis (rounds-to-target accuracy)
3. Per-round mean curves for plotting
4. Efficiency metrics aggregation
5. Client-level distributed metrics analysis
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SeedAggregationAnalyzer:
    def __init__(self, data_dir=".", output_dir="metrics/summary"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Method mapping (file prefix -> display name)
        self.methods = {
            'cfl': 'CFL',
            'fedprox': 'FedProx', 
            'hierfl': 'HierFL',
            'pfedme': 'pFedMe',
            'scaffold': 'SCAFFOLD',
            'fedge': 'FEDGE'
        }
        
        # Seeds available
        self.seeds = [0, 1]
        
        # Accuracy targets for convergence analysis
        self.accuracy_targets = [0.50, 0.60, 0.70, 0.75, 0.80]
        
    def calculate_confidence_interval(self, data, confidence=0.95):
        """Calculate confidence interval for given data."""
        if len(data) < 2:
            return np.nan, np.nan
        
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
        return mean - h, mean + h
    
    def load_centralized_metrics(self, method, seed):
        """Load centralized metrics for a specific method and seed."""
        filename = f"centralized_metrics_{method}_seed{seed}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
            
        df = pd.read_csv(filepath)
        return df
    
    def load_distributed_metrics(self, method, seed):
        """Load distributed metrics for a specific method and seed."""
        filename = f"distributed_metrics_{method}_seed{seed}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
            
        df = pd.read_csv(filepath)
        return df
    
    def load_fit_metrics(self, method, seed):
        """Load fit metrics for efficiency analysis."""
        filename = f"fit_metrics_{method}_seed{seed}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
            
        df = pd.read_csv(filepath)
        return df
    
    def analyze_final_performance(self):
        """Analyze final accuracy and loss across seeds for each method."""
        results = []
        
        for method_key, method_name in self.methods.items():
            final_accuracies = []
            final_losses = []
            last_rounds = []
            
            for seed in self.seeds:
                df = self.load_centralized_metrics(method_key, seed)
                if df is not None and len(df) > 0:
                    # Get final round metrics
                    final_row = df.iloc[-1]
                    final_accuracies.append(final_row['central_test_accuracy'])
                    final_losses.append(final_row['central_test_loss'])
                    last_rounds.append(final_row['round'])
            
            if final_accuracies:
                # Calculate statistics
                acc_mean = np.mean(final_accuracies)
                acc_std = np.std(final_accuracies, ddof=1) if len(final_accuracies) > 1 else 0
                acc_ci_lo, acc_ci_hi = self.calculate_confidence_interval(final_accuracies)
                
                loss_mean = np.mean(final_losses)
                loss_std = np.std(final_losses, ddof=1) if len(final_losses) > 1 else 0
                loss_ci_lo, loss_ci_hi = self.calculate_confidence_interval(final_losses)
                
                results.append({
                    'method': method_name,
                    'last_round': int(np.mean(last_rounds)),
                    'final_acc_mean': acc_mean,
                    'final_acc_std': acc_std,
                    'final_acc_ci95_lo': acc_ci_lo,
                    'final_acc_ci95_hi': acc_ci_hi,
                    'final_loss_mean': loss_mean,
                    'final_loss_std': loss_std,
                    'final_loss_ci95_lo': loss_ci_lo,
                    'final_loss_ci95_hi': loss_ci_hi
                })
        
        # Save results
        df_results = pd.DataFrame(results)
        output_path = self.output_dir / "seed_means.csv"
        df_results.to_csv(output_path, index=False)
        print(f"✅ Final performance analysis saved to {output_path}")
        
        return df_results
    
    def analyze_convergence(self):
        """Analyze convergence (rounds-to-target accuracy) across seeds."""
        results = []
        
        for method_key, method_name in self.methods.items():
            for target_acc in self.accuracy_targets:
                rounds_to_target = []
                
                for seed in self.seeds:
                    df = self.load_centralized_metrics(method_key, seed)
                    if df is not None and len(df) > 0:
                        # Find first round where accuracy >= target
                        target_rounds = df[df['central_test_accuracy'] >= target_acc]
                        if len(target_rounds) > 0:
                            rounds_to_target.append(target_rounds.iloc[0]['round'])
                        else:
                            # Target not reached
                            rounds_to_target.append(np.nan)
                
                if rounds_to_target and not all(np.isnan(rounds_to_target)):
                    # Filter out NaN values for statistics
                    valid_rounds = [r for r in rounds_to_target if not np.isnan(r)]
                    
                    if valid_rounds:
                        mean_rounds = np.mean(valid_rounds)
                        std_rounds = np.std(valid_rounds, ddof=1) if len(valid_rounds) > 1 else 0
                        ci_lo, ci_hi = self.calculate_confidence_interval(valid_rounds)
                        
                        results.append({
                            'method': method_name,
                            'target_accuracy': target_acc,
                            'mean_rounds': mean_rounds,
                            'std_rounds': std_rounds,
                            'ci95_lo': ci_lo,
                            'ci95_hi': ci_hi,
                            'seeds_reached': len(valid_rounds),
                            'total_seeds': len(self.seeds)
                        })
        
        # Save results
        df_results = pd.DataFrame(results)
        output_path = self.output_dir / "convergence.csv"
        df_results.to_csv(output_path, index=False)
        print(f"✅ Convergence analysis saved to {output_path}")
        
        return df_results
    
    def generate_per_round_curves(self):
        """Generate per-round mean curves across seeds for plotting."""
        curves_dir = self.output_dir / "curves"
        curves_dir.mkdir(exist_ok=True)
        
        for method_key, method_name in self.methods.items():
            # Collect data from all seeds
            seed_data = {}
            max_rounds = 0
            
            for seed in self.seeds:
                df = self.load_centralized_metrics(method_key, seed)
                if df is not None and len(df) > 0:
                    seed_data[seed] = df
                    max_rounds = max(max_rounds, df['round'].max())
            
            if not seed_data:
                continue
            
            # Create per-round aggregated data
            rounds_data = []
            
            for round_num in range(0, max_rounds + 1):
                round_accuracies = []
                round_losses = []
                
                for seed, df in seed_data.items():
                    round_row = df[df['round'] == round_num]
                    if len(round_row) > 0:
                        round_accuracies.append(round_row.iloc[0]['central_test_accuracy'])
                        round_losses.append(round_row.iloc[0]['central_test_loss'])
                
                if round_accuracies:
                    acc_mean = np.mean(round_accuracies)
                    acc_std = np.std(round_accuracies, ddof=1) if len(round_accuracies) > 1 else 0
                    acc_ci_lo, acc_ci_hi = self.calculate_confidence_interval(round_accuracies)
                    
                    loss_mean = np.mean(round_losses)
                    loss_std = np.std(round_losses, ddof=1) if len(round_losses) > 1 else 0
                    loss_ci_lo, loss_ci_hi = self.calculate_confidence_interval(round_losses)
                    
                    rounds_data.append({
                        'round': round_num,
                        'mean_accuracy': acc_mean,
                        'std_accuracy': acc_std,
                        'acc_ci95_lo': acc_ci_lo,
                        'acc_ci95_hi': acc_ci_hi,
                        'mean_loss': loss_mean,
                        'std_loss': loss_std,
                        'loss_ci95_lo': loss_ci_lo,
                        'loss_ci95_hi': loss_ci_hi
                    })
            
            # Save curve data
            if rounds_data:
                df_curve = pd.DataFrame(rounds_data)
                output_path = curves_dir / f"{method_name.lower()}_mean_curve.csv"
                df_curve.to_csv(output_path, index=False)
                print(f"✅ Curve data for {method_name} saved to {output_path}")
    
    def analyze_efficiency_metrics(self):
        """Analyze efficiency metrics across seeds."""
        results = []
        
        for method_key, method_name in self.methods.items():
            comp_times = []
            wall_times = []
            comm_mbs = []
            
            for seed in self.seeds:
                # Try distributed metrics first for communication data
                dist_df = self.load_distributed_metrics(method_key, seed)
                if dist_df is not None and len(dist_df) > 0:
                    # Get final round efficiency metrics
                    final_row = dist_df.iloc[-1]
                    
                    if 'total_comp_time_sec' in final_row:
                        comp_times.append(final_row['total_comp_time_sec'])
                    if 'total_wall_clock_sec' in final_row:
                        wall_times.append(final_row['total_wall_clock_sec'])
                    if 'total_communication_MB' in final_row:
                        comm_mbs.append(final_row['total_communication_MB'])
            
            # Calculate statistics if we have data
            if comp_times or wall_times or comm_mbs:
                result = {'method': method_name}
                
                if comp_times:
                    result.update({
                        'avg_comp_time_sec': np.mean(comp_times),
                        'std_comp_time_sec': np.std(comp_times, ddof=1) if len(comp_times) > 1 else 0
                    })
                
                if wall_times:
                    result.update({
                        'avg_wall_time_sec': np.mean(wall_times),
                        'std_wall_time_sec': np.std(wall_times, ddof=1) if len(wall_times) > 1 else 0
                    })
                
                if comm_mbs:
                    result.update({
                        'avg_communication_MB': np.mean(comm_mbs),
                        'std_communication_MB': np.std(comm_mbs, ddof=1) if len(comm_mbs) > 1 else 0
                    })
                
                results.append(result)
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            output_path = self.output_dir / "efficiency_metrics.csv"
            df_results.to_csv(output_path, index=False)
            print(f"✅ Efficiency metrics saved to {output_path}")
        
        return results
    
    def analyze_client_heterogeneity(self):
        """Analyze client-level distributed metrics for heterogeneity."""
        results = []
        
        for method_key, method_name in self.methods.items():
            final_client_acc_means = []
            final_client_acc_stds = []
            
            for seed in self.seeds:
                dist_df = self.load_distributed_metrics(method_key, seed)
                if dist_df is not None and len(dist_df) > 0:
                    final_row = dist_df.iloc[-1]
                    
                    # Get final round client heterogeneity
                    if 'avg_accuracy' in final_row:
                        final_client_acc_means.append(final_row['avg_accuracy'])
                    if 'accuracy_std' in final_row:
                        final_client_acc_stds.append(final_row['accuracy_std'])
            
            if final_client_acc_means:
                result = {
                    'method': method_name,
                    'client_acc_mean_across_seeds': np.mean(final_client_acc_means),
                    'client_acc_mean_std_across_seeds': np.std(final_client_acc_means, ddof=1) if len(final_client_acc_means) > 1 else 0
                }
                
                if final_client_acc_stds:
                    result.update({
                        'client_heterogeneity_mean': np.mean(final_client_acc_stds),
                        'client_heterogeneity_std': np.std(final_client_acc_stds, ddof=1) if len(final_client_acc_stds) > 1 else 0
                    })
                
                results.append(result)
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            output_path = self.output_dir / "client_heterogeneity.csv"
            df_results.to_csv(output_path, index=False)
            print(f"✅ Client heterogeneity analysis saved to {output_path}")
        
        return results
    
    def create_seed_level_summary(self):
        """Create per-seed raw results for sanity checking."""
        results = []
        
        for method_key, method_name in self.methods.items():
            for seed in self.seeds:
                df = self.load_centralized_metrics(method_key, seed)
                if df is not None and len(df) > 0:
                    final_row = df.iloc[-1]
                    results.append({
                        'method': method_name,
                        'seed': seed,
                        'last_round': final_row['round'],
                        'final_accuracy': final_row['central_test_accuracy'],
                        'final_loss': final_row['central_test_loss']
                    })
        
        # Save results
        df_results = pd.DataFrame(results)
        output_path = self.output_dir / "seed_level.csv"
        df_results.to_csv(output_path, index=False)
        print(f"✅ Seed-level summary saved to {output_path}")
        
        return df_results
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting comprehensive seed aggregation analysis...")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Output directory: {self.output_dir}")
        print(f"🔬 Methods: {list(self.methods.values())}")
        print(f"🌱 Seeds: {self.seeds}")
        print()
        
        # 1. Final performance analysis
        print("1️⃣ Analyzing final performance (accuracy & loss)...")
        final_perf = self.analyze_final_performance()
        
        # 2. Convergence analysis
        print("\n2️⃣ Analyzing convergence (rounds-to-target)...")
        convergence = self.analyze_convergence()
        
        # 3. Per-round curves
        print("\n3️⃣ Generating per-round mean curves...")
        self.generate_per_round_curves()
        
        # 4. Efficiency metrics
        print("\n4️⃣ Analyzing efficiency metrics...")
        efficiency = self.analyze_efficiency_metrics()
        
        # 5. Client heterogeneity
        print("\n5️⃣ Analyzing client heterogeneity...")
        heterogeneity = self.analyze_client_heterogeneity()
        
        # 6. Seed-level summary
        print("\n6️⃣ Creating seed-level summary...")
        seed_summary = self.create_seed_level_summary()
        
        print("\n✨ Analysis complete! Summary of results:")
        print(f"📈 Final Performance Summary ({len(final_perf)} methods):")
        for _, row in final_perf.iterrows():
            print(f"   {row['method']}: {row['final_acc_mean']:.4f} ± {row['final_acc_std']:.4f} accuracy")
        
        print(f"\n🎯 Convergence Analysis: {len(convergence)} target-method combinations")
        print(f"⚡ Efficiency Metrics: {len(efficiency)} methods analyzed")
        print(f"👥 Client Heterogeneity: {len(heterogeneity)} methods analyzed")
        
        return {
            'final_performance': final_perf,
            'convergence': convergence,
            'efficiency': efficiency,
            'heterogeneity': heterogeneity,
            'seed_summary': seed_summary
        }

def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = SeedAggregationAnalyzer(
        data_dir=".",  # Current directory contains the CSV files
        output_dir="metrics/summary"
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print(f"\n🎉 All analysis complete! Check the 'metrics/summary/' directory for results.")
    print("\n📋 Generated files:")
    print("   • seed_means.csv - Main results table (mean ± std ± CI)")
    print("   • convergence.csv - Rounds-to-target analysis")
    print("   • seed_level.csv - Per-seed raw results")
    print("   • efficiency_metrics.csv - Communication/computation metrics")
    print("   • client_heterogeneity.csv - Client variance analysis")
    print("   • curves/ - Per-round mean curves for plotting")

if __name__ == "__main__":
    main()
