"""
Plotting script for fine-tuning results

Generates comparison plots showing which fine-tuning approach works best.

USAGE:
    python plot_finetuning_results.py --root_path /path/to/repo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import argparse
import glob

parser = argparse.ArgumentParser(description="Plot fine-tuning results")
parser.add_argument('--root_path', type=str,
                   default="/Users/sylviaimanirakiza/Documents/GitHub/",
                   help="Root path of the project")
parser.add_argument('--results_file', type=str,
                   default="latest",
                   help="Specific results file or 'latest'")

args = parser.parse_args()
root_path = args.root_path

# Paths
results_path = f"{root_path}Generator_ML_Emission_Modeling/Modeling/summer25_results/cross_site_models/"
savepath = results_path

# Load results
if args.results_file == "latest":
    # Find most recent fine-tuning results file
    finetuning_files = glob.glob(f"{results_path}results_finetuning_*.csv")
    comparison_files = glob.glob(f"{results_path}results_comparison_*.csv")
    all_files = finetuning_files + comparison_files
    
    if not all_files:
        print("ERROR: No results found!")
        print(f"Looked in: {results_path}")
        print("Run fine_tune_from_saved_models.py first!")
        exit(1)
    
    results_file = max(all_files, key=os.path.getctime)
    print(f"Using latest results: {os.path.basename(results_file)}")
else:
    results_file = f"{results_path}{args.results_file}"

df = pd.read_csv(results_file)

print(f"\nLoaded {len(df)} results")
print(f"Sites: {df['site'].unique()}")
print(f"Approaches: {df['approach'].unique()}")
print(f"Days tested: {sorted(df['finetune_days'].unique())}")

# Plotting style
mpl.rcParams.update({
    'font.family': 'Arial',
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 20
})
sns.set_style("whitegrid")

# Colors for approaches
approach_colors = {
    'LOSO_baseline': '#808080',
    'continuation': '#2E86AB',
    'feature_adaptation': '#A23B72',
    'ensemble': '#F18F01'
}

approach_labels = {
    'LOSO_baseline': 'LOSO Baseline',
    'continuation': 'Continuation',
    'feature_adaptation': 'Feature Adaptation',
    'ensemble': 'Weighted Ensemble'
}

# ============================================================================
# PLOT 1: Accuracy Learning Curves (2x2 grid)
# ============================================================================

sites = sorted(df['site'].unique())
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, site in enumerate(sites):
    ax = axes[idx]
    site_data = df[df['site'] == site].copy()
    
    # Plot each approach
    approaches = [a for a in site_data['approach'].unique() if a != 'LOSO_baseline']
    
    for approach_name in approaches:
        approach_data = site_data[site_data['approach'] == approach_name].sort_values('finetune_days')
        
        if len(approach_data) > 0:
            ax.plot(approach_data['finetune_days'], 
                   approach_data['accuracy'],
                   marker='o', linewidth=3, markersize=10,
                   label=approach_labels.get(approach_name, approach_name),
                   color=approach_colors.get(approach_name, None))
    
    # Add LOSO baseline
    baseline = site_data[site_data['approach'] == 'LOSO_baseline']
    if not baseline.empty:
        ax.axhline(y=baseline['accuracy'].values[0],
                  color=approach_colors['LOSO_baseline'],
                  linestyle='--', linewidth=2.5,
                  label='LOSO Baseline', alpha=0.7)
    
    ax.set_title(f'{site}', fontweight='bold', fontsize=18)
    ax.set_xlabel('Fine-tuning Days', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

plt.suptitle('Fine-Tuning Comparison: Accuracy vs Days',
            fontweight='bold', fontsize=22, y=0.995)
plt.tight_layout()
plt.savefig(f'{savepath}/finetuning_accuracy_by_site.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: finetuning_accuracy_by_site.png")
plt.show()

# ============================================================================
# PLOT 2: Improvement Bars
# ============================================================================

days_tested = sorted([d for d in df['finetune_days'].unique() if d > 0])

if len(days_tested) > 0:
    fig, axes = plt.subplots(1, len(days_tested), 
                            figsize=(6*len(days_tested), 8),
                            sharey=True)
    
    if len(days_tested) == 1:
        axes = [axes]
    
    for idx, days in enumerate(days_tested):
        ax = axes[idx]
        
        # Calculate improvements
        improvements = []
        for site in sites:
            site_data = df[df['site'] == site]
            baseline = site_data[site_data['approach'] == 'LOSO_baseline']['accuracy'].values[0]
            
            day_data = site_data[site_data['finetune_days'] == days]
            
            for _, row in day_data.iterrows():
                improvements.append({
                    'site': site,
                    'approach': row['approach'],
                    'improvement': (row['accuracy'] - baseline) * 100
                })
        
        imp_df = pd.DataFrame(improvements)
        pivot_imp = imp_df.pivot(index='site', columns='approach', values='improvement')
        
        # Plot
        pivot_imp.plot(kind='bar', ax=ax, width=0.8,
                      color=[approach_colors.get(col, None) for col in pivot_imp.columns])
        
        ax.set_title(f'{int(days)} Day(s)', fontweight='bold', fontsize=18)
        ax.set_xlabel('Site', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Accuracy Improvement (pp)', fontweight='bold', fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.legend(title='Approach', labels=[approach_labels.get(col, col) for col in pivot_imp.columns])
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Improvement over LOSO Baseline',
                fontweight='bold', fontsize=22, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{savepath}/finetuning_improvement_bars.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: finetuning_improvement_bars.png")
    plt.show()

# ============================================================================
# PLOT 3: Best Approach Heatmap
# ============================================================================

if len(days_tested) > 0:
    best_approach_matrix = pd.DataFrame(index=sites, columns=[f"{int(d)}d" for d in days_tested])
    improvement_matrix = pd.DataFrame(index=sites, columns=[f"{int(d)}d" for d in days_tested])
    
    for site in sites:
        site_data = df[df['site'] == site]
        baseline_acc = site_data[site_data['approach'] == 'LOSO_baseline']['accuracy'].values[0]
        
        for days in days_tested:
            day_data = site_data[site_data['finetune_days'] == days]
            if not day_data.empty:
                best_row = day_data.loc[day_data['accuracy'].idxmax()]
                best_approach_matrix.loc[site, f"{int(days)}d"] = best_row['approach']
                improvement_matrix.loc[site, f"{int(days)}d"] = (best_row['accuracy'] - baseline_acc) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap 1: Best approach
    approach_to_num = {name: i for i, name in enumerate(df['approach'].unique()) if name != 'LOSO_baseline'}
    numeric_matrix = best_approach_matrix.replace(approach_to_num)
    
    sns.heatmap(numeric_matrix.astype(float), annot=best_approach_matrix, fmt='',
               cmap='Set3', ax=ax1, cbar=False, linewidths=1, linecolor='black')
    ax1.set_title('Best Approach per Site', fontweight='bold', fontsize=18)
    ax1.set_xlabel('Fine-tuning Days', fontweight='bold')
    ax1.set_ylabel('Site', fontweight='bold')
    
    # Heatmap 2: Improvement magnitude
    sns.heatmap(improvement_matrix.astype(float), annot=True, fmt='.1f',
               cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Improvement (pp)'},
               linewidths=1, linecolor='black', vmin=0, vmax=20)
    ax2.set_title('Accuracy Improvement (pp)', fontweight='bold', fontsize=18)
    ax2.set_xlabel('Fine-tuning Days', fontweight='bold')
    ax2.set_ylabel('Site', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{savepath}/finetuning_best_approach_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: finetuning_best_approach_heatmap.png")
    plt.show()

# ============================================================================
# PLOT 4: F1 Score Comparison
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, site in enumerate(sites):
    ax = axes[idx]
    site_data = df[df['site'] == site].copy()
    
    approaches = [a for a in site_data['approach'].unique() if a != 'LOSO_baseline']
    
    for approach_name in approaches:
        approach_data = site_data[site_data['approach'] == approach_name].sort_values('finetune_days')
        
        if len(approach_data) > 0:
            ax.plot(approach_data['finetune_days'],
                   approach_data['f1'],
                   marker='s', linewidth=3, markersize=10,
                   label=approach_labels.get(approach_name, approach_name),
                   color=approach_colors.get(approach_name, None))
    
    baseline = site_data[site_data['approach'] == 'LOSO_baseline']
    if not baseline.empty:
        ax.axhline(y=baseline['f1'].values[0],
                  color=approach_colors['LOSO_baseline'],
                  linestyle='--', linewidth=2.5,
                  label='LOSO Baseline', alpha=0.7)
    
    ax.set_title(f'{site}', fontweight='bold', fontsize=18)
    ax.set_xlabel('Fine-tuning Days', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

plt.suptitle('Fine-Tuning Comparison: F1 Score vs Days',
            fontweight='bold', fontsize=22, y=0.995)
plt.tight_layout()
plt.savefig(f'{savepath}/finetuning_f1_by_site.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: finetuning_f1_by_site.png")
plt.show()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*100)
print("APPROACH RANKING BY PERFORMANCE")
print("="*100)

for days in days_tested:
    print(f"\n{int(days)} Day(s):")
    print(f"{'-'*50}")
    
    day_data = df[df['finetune_days'] == days]
    approach_improvements = {}
    
    for approach_name in day_data['approach'].unique():
        approach_data = day_data[day_data['approach'] == approach_name]
        
        improvements = []
        for site in sites:
            site_baseline = df[(df['site'] == site) & (df['approach'] == 'LOSO_baseline')]['accuracy'].values[0]
            site_approach = approach_data[approach_data['site'] == site]
            
            if not site_approach.empty:
                imp = (site_approach['accuracy'].values[0] - site_baseline) * 100
                improvements.append(imp)
        
        if improvements:
            approach_improvements[approach_name] = np.mean(improvements)
    
    ranked = sorted(approach_improvements.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (approach_name, avg_imp) in enumerate(ranked, 1):
        print(f"  {rank}. {approach_labels.get(approach_name, approach_name):<25} "
              f"Avg improvement: {avg_imp:+.2f} pp")

# Save summary
summary_data = []
for days in days_tested:
    day_data = df[df['finetune_days'] == days]
    
    for approach_name in day_data['approach'].unique():
        approach_data = day_data[day_data['approach'] == approach_name]
        
        for site in sites:
            site_baseline = df[(df['site'] == site) & (df['approach'] == 'LOSO_baseline')]['accuracy'].values[0]
            site_approach = approach_data[approach_data['site'] == site]
            
            if not site_approach.empty:
                summary_data.append({
                    'days': int(days),
                    'site': site,
                    'approach': approach_labels.get(approach_name, approach_name),
                    'accuracy': site_approach['accuracy'].values[0],
                    'f1': site_approach['f1'].values[0],
                    'improvement_pp': (site_approach['accuracy'].values[0] - site_baseline) * 100
                })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{savepath}/finetuning_summary_table.csv', index=False)
print(f"\n✓ Summary table saved to: finetuning_summary_table.csv")

print("\n" + "="*100)
print("ALL PLOTS GENERATED!")
print("="*100)
print(f"\nGenerated files:")
print(f"  1. finetuning_accuracy_by_site.png")
print(f"  2. finetuning_improvement_bars.png")
print(f"  3. finetuning_best_approach_heatmap.png")
print(f"  4. finetuning_f1_by_site.png")
print(f"  5. finetuning_summary_table.csv")
print("="*100 + "\n")
