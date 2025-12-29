#!/usr/bin/env python3
"""
Clean hierarchical federated learning results plotter for CIFAR-10.
Generates focused plots for server accuracy, loss, and performance analysis.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class HierarchicalResultsAnalyzer:
    """Clean analyzer for hierarchical FL results"""
    
    def __init__(self, rounds_dir="./rounds", runs_dir="./runs"):
        self.rounds_dir = Path(rounds_dir)
        self.runs_dir = Path(runs_dir)
        self.round_data = {}
        self.available_rounds = []
        
        # Create plots directory
        self.output_dir = Path("plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Detect available rounds
        self._detect_rounds()
    
    def _detect_rounds(self):
        """Detect available rounds from directory structure"""
        round_numbers = []
        if self.rounds_dir.exists():
            for round_dir in self.rounds_dir.glob("round_*"):
                try:
                    round_num = int(round_dir.name.split('_')[1])
                    if (round_dir / "leaf").exists():
                        round_numbers.append(round_num)
                except (ValueError, IndexError):
                    continue
        
        self.available_rounds = sorted(round_numbers)
        print(f"🔍 Found {len(self.available_rounds)} rounds: {min(round_numbers) if round_numbers else 0}-{max(round_numbers) if round_numbers else 0}")
    
    def load_metrics(self):
        """Load server metrics from CSV files"""
        print("📊 Loading metrics...")
        
        for round_num in self.available_rounds:
            self.round_data[round_num] = {
                'server_accuracies': {},
                'server_losses': {},
                'global_accuracy': None,
                'global_loss': None
            }
            
            # Load server metrics for each server
            for server_id in range(3):
                server_dir = self.rounds_dir / f"round_{round_num}" / "leaf" / f"server_{server_id}"
                metrics_file = server_dir / "server_metrics.csv"
                
                if metrics_file.exists():
                    try:
                        df = pd.read_csv(metrics_file)
                        if not df.empty:
                            row = df.iloc[0]
                            self.round_data[round_num]['server_accuracies'][server_id] = row['agg_acc']
                            self.round_data[round_num]['server_losses'][server_id] = row['agg_loss']
                    except Exception as e:
                        print(f"⚠️ Error loading server {server_id} metrics for round {round_num}: {e}")
        
        # Load global metrics from runs directory
        self._load_global_metrics()
        print(f"✅ Loaded metrics for {len(self.round_data)} rounds")
    
    def _load_global_metrics(self):
        """Load global model metrics from runs directory"""
        if not self.runs_dir.exists():
            return
        
        run_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return
        
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        global_metrics_file = latest_run / "global_model_metrics.csv"
        
        if global_metrics_file.exists():
            try:
                df_global = pd.read_csv(global_metrics_file)
                for _, row in df_global.iterrows():
                    round_num = int(row['global_round'])
                    if round_num in self.round_data:
                        self.round_data[round_num]['global_accuracy'] = row['accuracy']
                        self.round_data[round_num]['global_loss'] = row['loss']
                print(f"✅ Loaded global metrics from {latest_run.name}")
            except Exception as e:
                print(f"⚠️ Error loading global metrics: {e}")
    
    def plot_server_accuracy(self):
        """Plot server test accuracy vs rounds"""
        if not self.round_data:
            print("⚠️ No data available for plotting")
            return
        
        rounds = sorted(self.round_data.keys())
        
        # Extract server accuracies
        server_0_accs = [self.round_data[r]['server_accuracies'].get(0, 0.0) for r in rounds]
        server_1_accs = [self.round_data[r]['server_accuracies'].get(1, 0.0) for r in rounds]
        server_2_accs = [self.round_data[r]['server_accuracies'].get(2, 0.0) for r in rounds]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(rounds, server_0_accs, 'b-', linewidth=2, marker='o', markersize=4, label='Server 0')
        ax.plot(rounds, server_1_accs, 'r-', linewidth=2, marker='s', markersize=4, label='Server 1')
        ax.plot(rounds, server_2_accs, 'g-', linewidth=2, marker='^', markersize=4, label='Server 2')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Server Test Accuracy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_xticks(range(0, max(rounds)+1, 10) if rounds else range(0, 151, 10))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'server_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Server accuracy plot saved")
    
    def plot_server_loss(self):
        """Plot server test loss vs rounds"""
        if not self.round_data:
            print("⚠️ No data available for plotting")
            return
        
        rounds = sorted(self.round_data.keys())
        
        # Extract server losses
        server_0_losses = [self.round_data[r]['server_losses'].get(0, 2.0) for r in rounds]
        server_1_losses = [self.round_data[r]['server_losses'].get(1, 2.0) for r in rounds]
        server_2_losses = [self.round_data[r]['server_losses'].get(2, 2.0) for r in rounds]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(rounds, server_0_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Server 0')
        ax.plot(rounds, server_1_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Server 1')
        ax.plot(rounds, server_2_losses, 'g-', linewidth=2, marker='^', markersize=4, label='Server 2')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Test Loss')
        ax.set_title('Server Test Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yticks(np.arange(0, 4.5, 0.5))
        ax.set_xticks(range(0, max(rounds)+1, 10) if rounds else range(0, 151, 10))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'server_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Server loss plot saved")
    
    def plot_individual_servers(self):
        """Create individual accuracy/loss plots for each server"""
        if not self.round_data:
            print("⚠️ No data available for plotting")
            return
        
        rounds = sorted(self.round_data.keys())
        
        for server_id in range(3):
            server_accs = [self.round_data[r]['server_accuracies'].get(server_id, 0.0) for r in rounds]
            server_losses = [self.round_data[r]['server_losses'].get(server_id, 2.0) for r in rounds]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Server {server_id} Performance', fontsize=16, fontweight='bold')
            
            # Accuracy plot
            ax1.plot(rounds, server_accs, 'b-', linewidth=2, marker='o', markersize=3)
            ax1.set_xlabel('Training Round')
            ax1.set_ylabel('Test Accuracy')
            ax1.set_title(f'Server {server_id} Test Accuracy')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.set_yticks(np.arange(0, 1.1, 0.2))
            ax1.set_xticks(range(0, max(rounds)+1, 10) if rounds else range(0, 151, 10))
            
            # Loss plot
            ax2.plot(rounds, server_losses, 'r-', linewidth=2, marker='s', markersize=3)
            ax2.set_xlabel('Training Round')
            ax2.set_ylabel('Test Loss')
            ax2.set_title(f'Server {server_id} Test Loss')
            ax2.grid(True, alpha=0.3)
            ax2.set_yticks(np.arange(0, 4.5, 0.5))
            ax2.set_xticks(range(0, max(rounds)+1, 10) if rounds else range(0, 151, 10))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'server_{server_id}_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Server {server_id} individual plot saved")
    
    def plot_global_vs_servers(self):
        """Plot global model vs server performance comparison"""
        if not self.round_data:
            print("⚠️ No data available for plotting")
            return
        
        rounds = sorted(self.round_data.keys())
        
        # Extract data
        global_accs = []
        server_avg_accs = []
        
        for r in rounds:
            global_acc = self.round_data[r].get('global_accuracy')
            server_accs = list(self.round_data[r]['server_accuracies'].values())
            
            if global_acc is not None:
                global_accs.append(global_acc)
            else:
                global_accs.append(np.mean(server_accs) if server_accs else 0.0)
            
            server_avg_accs.append(np.mean(server_accs) if server_accs else 0.0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(rounds, global_accs, 'darkblue', linewidth=3, marker='o', markersize=5, label='Global Model')
        ax.plot(rounds, server_avg_accs, 'lightblue', linewidth=2, marker='s', markersize=4, label='Server Average')
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Global Model vs Server Average Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_xticks(range(0, max(rounds)+1, 10) if rounds else range(0, 151, 10))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'global_vs_servers.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Global vs servers plot saved")
    
    def generate_report(self):
        """Generate summary report"""
        if not self.round_data:
            print("⚠️ No data available for report")
            return
        
        final_round = max(self.available_rounds)
        final_data = self.round_data[final_round]
        
        print("\n" + "="*60)
        print(f"HIERARCHICAL FL ANALYSIS REPORT - {len(self.available_rounds)} ROUNDS")
        print("="*60)
        
        print(f"\n📊 FINAL PERFORMANCE (Round {final_round}):")
        for server_id in range(3):
            acc = final_data['server_accuracies'].get(server_id, 0.0)
            loss = final_data['server_losses'].get(server_id, 0.0)
            print(f"   • Server {server_id}: Accuracy={acc:.3f}, Loss={loss:.3f}")
        
        global_acc = final_data.get('global_accuracy')
        if global_acc:
            print(f"   • Global Model: Accuracy={global_acc:.3f}")
        
        print("="*60)
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Clean Hierarchical FL Analysis...")
        
        if not self.available_rounds:
            print("❌ No rounds found for analysis")
            return
        
        # Load data
        self.load_metrics()
        
        # Generate plots
        print("\n📈 Generating plots...")
        self.plot_server_accuracy()
        self.plot_server_loss()
        self.plot_individual_servers()
        self.plot_global_vs_servers()
        
        # Generate report
        self.generate_report()
        
        print(f"\n✅ Analysis complete! Plots saved to: {self.output_dir.resolve()}")
        print(f"\n📁 Generated Files:")
        print(f"   • server_accuracy.png - Server accuracy evolution")
        print(f"   • server_loss.png - Server loss evolution")
        print(f"   • server_0_performance.png - Server 0 individual plots")
        print(f"   • server_1_performance.png - Server 1 individual plots")
        print(f"   • server_2_performance.png - Server 2 individual plots")
        print(f"   • global_vs_servers.png - Global vs server comparison")

if __name__ == "__main__":
    analyzer = HierarchicalResultsAnalyzer()
    analyzer.run_analysis()
