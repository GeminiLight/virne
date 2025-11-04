#!/usr/bin/env python3
"""
Standalone script to run D-Rounding experiments on tree topology with 5 different seeds.

This script runs the D-Rounding solver on a tree topology with clean switches (CPU=0)
for 200 VNR requests across 5 different random seeds for statistical significance.

Usage:
    python run_drounding_tree_experiment.py

Alternative: Use the bash script for more control
    ./run_drounding_tree_5seeds.sh
"""

import os
import sys
import subprocess
from pathlib import Path


def run_experiment(seed):
    """
    Run a single D-Rounding experiment with the given seed.

    Args:
        seed: Random seed for the experiment

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running D-Rounding Tree Experiment - Seed {seed}")
    print(f"{'='*60}\n")

    cmd = [
        "python",
        "main_tree_drounding.py",
        "solver.solver_name=d_round",
        f"experiment.seed={seed}",
        f"experiment.run_id=drounding_tree_seed_{seed}"
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Seed {seed} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Seed {seed} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Seed {seed} failed with exception: {e}")
        return False


def main():
    """
    Main function to run all experiments.
    """
    print("\n" + "="*60)
    print("D-Rounding Tree Topology Experiment Suite")
    print("="*60)
    print("\nConfiguration:")
    print("  - Solver: D-Rounding (Deterministic Rounding)")
    print("  - Topology: Tree with binary branching")
    print("  - Switches: Routing-only (CPU=0)")
    print("  - Hosts: 16 nodes with CPU resources")
    print("  - VNRs: 200 requests per seed")
    print("  - Seeds: 0, 1, 2, 3, 4")
    print("="*60 + "\n")

    # Check if we're in the correct directory
    if not Path("main_tree_drounding.py").exists():
        print("Error: main_tree_drounding.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    # Seeds to run
    seeds = [0, 1, 2, 3, 4]

    # Track results
    successful_seeds = []
    failed_seeds = []

    # Run experiments for each seed
    for seed in seeds:
        success = run_experiment(seed)
        if success:
            successful_seeds.append(seed)
        else:
            failed_seeds.append(seed)

    # Print summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"\nTotal seeds: {len(seeds)}")
    print(f"Successful: {len(successful_seeds)} - {successful_seeds}")
    print(f"Failed: {len(failed_seeds)} - {failed_seeds}")

    if len(failed_seeds) == 0:
        print("\n✓ All experiments completed successfully!")
        print("\nResults saved in:")
        print("  virne/d_round/drounding_tree_seed_*/records/")
        print("\nNext steps:")
        print("  1. Analyze results across all seeds")
        print("  2. Compare acceptance rates and R2C ratios")
        print("  3. Compare with MIP solver results")
    else:
        print("\n✗ Some experiments failed. Please check the logs.")
        sys.exit(1)

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
