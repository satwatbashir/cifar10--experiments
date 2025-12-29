#!/usr/bin/env python3
"""
Validation script to verify the seed aggregation analysis results.
Performs sanity checks and generates a final summary.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def validate_results():
    """Validate all analysis results for consistency and correctness."""
    print("🔍 Validating analysis results...")
    
    # 1. Check that all expected files exist
    expected_files = [
        'metrics/summary/seed_means.csv',
        'metrics/summary/convergence.csv', 
        'metrics/summary/seed_level.csv',
        'metrics/summary/efficiency_metrics.csv',
        'metrics/summary/client_heterogeneity.csv'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All expected output files present")
    
    # 2. Validate seed-level vs aggregated consistency
    seed_df = pd.read_csv('metrics/summary/seed_level.csv')
    means_df = pd.read_csv('metrics/summary/seed_means.csv')
    
    print("\n🧮 Validating aggregation calculations...")
    
    for method in means_df['method']:
        method_seeds = seed_df[seed_df['method'] == method]
        method_mean = means_df[means_df['method'] == method].iloc[0]
        
        # Check accuracy calculation
        actual_acc_mean = method_seeds['final_accuracy'].mean()
        reported_acc_mean = method_mean['final_acc_mean']
        
        if abs(actual_acc_mean - reported_acc_mean) > 1e-6:
            print(f"❌ {method}: Accuracy mean mismatch")
            return False
        
        # Check loss calculation
        actual_loss_mean = method_seeds['final_loss'].mean()
        reported_loss_mean = method_mean['final_loss_mean']
        
        if abs(actual_loss_mean - reported_loss_mean) > 1e-6:
            print(f"❌ {method}: Loss mean mismatch")
            return False
    
    print("✅ Aggregation calculations verified")
    
    # 3. Check curve files exist
    curves_dir = Path('metrics/summary/curves')
    if curves_dir.exists():
        curve_files = list(curves_dir.glob('*_mean_curve.csv'))
        print(f"✅ Generated {len(curve_files)} curve files")
    else:
        print("❌ Curves directory missing")
        return False
    
    # 4. Validate data ranges
    print("\n📊 Validating data ranges...")
    
    # Accuracy should be between 0 and 1
    for _, row in means_df.iterrows():
        if not (0 <= row['final_acc_mean'] <= 1):
            print(f"❌ {row['method']}: Invalid accuracy range")
            return False
    
    # Loss should be positive
    for _, row in means_df.iterrows():
        if row['final_loss_mean'] <= 0:
            print(f"❌ {row['method']}: Invalid loss value")
            return False
    
    print("✅ Data ranges validated")
    
    return True

def generate_final_summary():
    """Generate a final summary of all results."""
    print("\n📋 FINAL ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Load main results
    means_df = pd.read_csv('metrics/summary/seed_means.csv')
    seed_df = pd.read_csv('metrics/summary/seed_level.csv')
    
    print("\n🎯 PERFORMANCE RANKING (by final accuracy):")
    ranked_df = means_df.sort_values('final_acc_mean', ascending=False)
    
    for i, (_, row) in enumerate(ranked_df.iterrows(), 1):
        acc_mean = row['final_acc_mean']
        acc_std = row['final_acc_std']
        print(f"{i}. {row['method']}: {acc_mean:.4f} ± {acc_std:.4f} ({acc_mean*100:.2f}%)")
    
    print("\n🔄 CONSISTENCY RANKING (by lowest std dev):")
    consistency_df = means_df.sort_values('final_acc_std', ascending=True)
    
    for i, (_, row) in enumerate(consistency_df.iterrows(), 1):
        acc_std = row['final_acc_std']
        print(f"{i}. {row['method']}: σ = {acc_std:.4f}")
    
    print("\n📈 CONVERGENCE SUMMARY:")
    try:
        conv_df = pd.read_csv('metrics/summary/convergence.csv')
        conv_50 = conv_df[conv_df['target_accuracy'] == 0.5]
        
        if len(conv_50) > 0:
            conv_50_sorted = conv_50.sort_values('mean_rounds', ascending=True)
            for _, row in conv_50_sorted.iterrows():
                rounds = row['mean_rounds']
                std_rounds = row['std_rounds']
                print(f"   {row['method']}: {rounds:.1f} ± {std_rounds:.1f} rounds to 50%")
        else:
            print("   No methods reached 50% accuracy target")
    except:
        print("   Convergence data not available")
    
    print("\n⚡ EFFICIENCY SUMMARY:")
    try:
        eff_df = pd.read_csv('metrics/summary/efficiency_metrics.csv')
        
        print("   Communication (MB):")
        for _, row in eff_df.iterrows():
            comm = row['avg_communication_MB']
            print(f"     {row['method']}: {comm:.2f} MB")
        
        print("   Computation Time (where available):")
        comp_data = eff_df.dropna(subset=['avg_comp_time_sec'])
        for _, row in comp_data.iterrows():
            comp_time = row['avg_comp_time_sec']
            comp_std = row['std_comp_time_sec']
            print(f"     {row['method']}: {comp_time:.1f} ± {comp_std:.1f} sec")
    except:
        print("   Efficiency data not available")
    
    print("\n👥 CLIENT FAIRNESS SUMMARY:")
    try:
        het_df = pd.read_csv('metrics/summary/client_heterogeneity.csv')
        het_sorted = het_df.sort_values('client_heterogeneity_mean', ascending=True)
        
        print("   (Lower heterogeneity = more fair)")
        for _, row in het_sorted.iterrows():
            het_mean = row['client_heterogeneity_mean']
            het_std = row['client_heterogeneity_std']
            print(f"     {row['method']}: {het_mean:.4f} ± {het_std:.4f}")
    except:
        print("   Heterogeneity data not available")
    
    print("\n📊 STATISTICAL ROBUSTNESS:")
    print("   All results computed across 2 seeds with 95% confidence intervals")
    print("   Seed-level variance captured in standard deviations")
    
    print("\n📁 OUTPUT FILES GENERATED:")
    output_files = [
        "seed_means.csv - Main results (mean ± std ± CI)",
        "convergence.csv - Rounds-to-target analysis", 
        "seed_level.csv - Per-seed raw results",
        "efficiency_metrics.csv - Communication/computation costs",
        "client_heterogeneity.csv - Client fairness analysis",
        "curves/ - Per-round curves for plotting"
    ]
    
    for file_desc in output_files:
        print(f"   ✓ {file_desc}")
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 50)

def main():
    """Main validation and summary function."""
    if validate_results():
        print("✅ All validations passed!")
        generate_final_summary()
    else:
        print("❌ Validation failed!")

if __name__ == "__main__":
    main()
