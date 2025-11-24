#!/usr/bin/env python3
"""
Analyze simulation results and create boxplots for acceptance rate and solution time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

def load_all_summaries():
    """Load all summary CSV files from simulation results."""
    summaries = []

    # Find all summary CSV files
    pattern = "apresentacao/simulacoes/**/*summary*.csv"
    files = glob.glob(pattern, recursive=True)

    print(f"Found {len(files)} summary files")

    for file in files:
        try:
            # Extract algorithm name from path
            parts = file.split('/')
            algo_dir = parts[2]  # apresentacao/simulacoes/[algo]/...

            # Map directory names to readable algorithm names
            algo_map = {
                'ga_meta': 'GA',
                'mip': 'MIP',
                'mcts': 'MCTS',
                'sa_meta': 'SA',
                'pso_meta': 'PSO',
                'pl_rank': 'PL-Rank',
                'rw_rank_bfs': 'RW-Rank-BFS',
                'd_round': 'D-Rounding'
            }

            algo_name = algo_map.get(algo_dir, algo_dir)

            # Extract topology from filename
            if 'tree' in file.lower() and 'fat' not in file.lower():
                topology = 'Tree'
            elif 'fat' in file.lower():
                topology = 'Fat-Tree'
            else:
                topology = 'Unknown'

            # Read CSV
            df = pd.read_csv(file)

            # Check if this is a summary file with the metrics we need
            if 'clock_running_time' in df.columns or 'acceptance_rate' in df.columns:
                # Get metrics (assuming one row per simulation)
                for idx, row in df.iterrows():
                    # Use actual wall-clock time, not simulation time
                    clock_time = row.get('clock_running_time', None)
                    num_vnrs = 200  # Fixed number of VNRs

                    # Calculate average time per VNR (in seconds)
                    avg_time_per_vnr = clock_time / num_vnrs if clock_time else None

                    summaries.append({
                        'algorithm': algo_name,
                        'topology': topology,
                        'file': os.path.basename(file),
                        'acceptance_rate': row.get('acceptance_rate', None),
                        'avg_time_per_vnr': avg_time_per_vnr,  # Average seconds per VNR
                        'total_run_time': clock_time,  # Total wall-clock time
                        'r2c_ratio': row.get('avg_r2c_ratio', None),
                        'num_requests': num_vnrs,
                        'success_count': row.get('success_count', None)
                    })

        except Exception as e:
            print(f"Error reading {file}: {e}")

    return pd.DataFrame(summaries)

def check_missing_data(df):
    """Check for missing or invalid data."""
    print("\n" + "="*80)
    print("DATA QUALITY CHECK")
    print("="*80)

    print(f"\nTotal records: {len(df)}")
    print(f"\nRecords per algorithm:")
    print(df['algorithm'].value_counts().sort_index())

    print(f"\n\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # Check for NaN or infinite values
    print(f"\n\nNaN values in acceptance_rate: {df['acceptance_rate'].isna().sum()}")
    print(f"NaN values in avg_time_per_vnr: {df['avg_time_per_vnr'].isna().sum()}")

    # Check for extreme values (potential timeouts)
    if not df['avg_time_per_vnr'].isna().all():
        print(f"\n\nAverage time per VNR statistics (seconds):")
        print(df['avg_time_per_vnr'].describe())

        print(f"\n\nTotal run time statistics:")
        print(df['total_run_time'].describe())

        # Flag potential timeouts (very high times)
        max_time = df['avg_time_per_vnr'].max()
        if max_time > 60:  # More than 1 minute per VNR
            print(f"\n⚠️  WARNING: Found very high per-VNR times (max: {max_time:.2f}s)")
            high_time = df[df['avg_time_per_vnr'] > 60]
            print(f"   {len(high_time)} records with time > 60s per VNR")
            print(high_time[['algorithm', 'topology', 'avg_time_per_vnr']])

    print("\n" + "="*80)

    return df

def create_boxplots(df):
    """Create boxplots for acceptance rate and solution time."""

    # Remove rows with missing data for plotting
    df_clean = df.dropna(subset=['acceptance_rate', 'avg_time_per_vnr'])

    print(f"\nRecords with complete data: {len(df_clean)} / {len(df)}")

    if len(df_clean) == 0:
        print("ERROR: No complete data available for plotting!")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sort algorithms for consistent display
    algo_order = sorted(df_clean['algorithm'].unique())

    # Boxplot 1: Acceptance Rate
    sns.boxplot(data=df_clean, x='algorithm', y='acceptance_rate',
                order=algo_order, ax=ax1, palette='Set2')
    ax1.set_title('Acceptance Rate by Algorithm', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add median values as text
    medians = df_clean.groupby('algorithm')['acceptance_rate'].median()
    for i, algo in enumerate(algo_order):
        if algo in medians.index:
            ax1.text(i, medians[algo], f'{medians[algo]:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Boxplot 2: Average Time per VNR (log scale if needed)
    sns.boxplot(data=df_clean, x='algorithm', y='avg_time_per_vnr',
                order=algo_order, ax=ax2, palette='Set2')
    ax2.set_title('Average Time per VNR by Algorithm', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Average Time per VNR (seconds)', fontsize=12)

    # Use log scale if there's a large range
    time_range = df_clean['avg_time_per_vnr'].max() / df_clean['avg_time_per_vnr'].min()
    if time_range > 100:
        ax2.set_yscale('log')
        ax2.set_ylabel('Average Time per VNR (seconds, log scale)', fontsize=12)

    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add median values as text
    medians_time = df_clean.groupby('algorithm')['avg_time_per_vnr'].median()
    for i, algo in enumerate(algo_order):
        if algo in medians_time.index:
            ax2.text(i, medians_time[algo], f'{medians_time[algo]:.2f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_file = 'apresentacao/simulacoes/results_boxplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Boxplots saved to: {output_file}")

    plt.close()

    # Create separate plots by topology if we have both
    if len(df_clean['topology'].unique()) > 1:
        create_topology_comparison(df_clean)

def create_topology_comparison(df):
    """Create separate comparison for different topologies."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, topology in enumerate(['Tree', 'Fat-Tree']):
        df_topo = df[df['topology'] == topology]

        if len(df_topo) == 0:
            continue

        algo_order = sorted(df_topo['algorithm'].unique())

        # Acceptance Rate
        sns.boxplot(data=df_topo, x='algorithm', y='acceptance_rate',
                   order=algo_order, ax=axes[i, 0], palette='Set2')
        axes[i, 0].set_title(f'{topology} - Acceptance Rate', fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('Algorithm', fontsize=10)
        axes[i, 0].set_ylabel('Acceptance Rate (%)', fontsize=10)
        axes[i, 0].set_ylim([0, 1.0])
        plt.setp(axes[i, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Average Time per VNR
        sns.boxplot(data=df_topo, x='algorithm', y='avg_time_per_vnr',
                   order=algo_order, ax=axes[i, 1], palette='Set2')
        axes[i, 1].set_title(f'{topology} - Avg Time per VNR', fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('Algorithm', fontsize=10)
        axes[i, 1].set_ylabel('Avg Time per VNR (seconds)', fontsize=10)
        plt.setp(axes[i, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    output_file = 'apresentacao/simulacoes/results_by_topology.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Topology comparison saved to: {output_file}")

    plt.close()

def main():
    print("Analyzing simulation results...")

    # Load all data
    df = load_all_summaries()

    if len(df) == 0:
        print("ERROR: No summary data found!")
        return

    # Check data quality
    df = check_missing_data(df)

    # Save raw data
    output_csv = 'apresentacao/simulacoes/all_results_summary.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✓ All results saved to: {output_csv}")

    # Create boxplots
    create_boxplots(df)

    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
