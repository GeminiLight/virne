"""
Main script for running MIP experiments on Fat-Tree datacenter topology.

Usage:
    python main_fat_tree_mip.py experiment.seed=0
"""

import hydra
from omegaconf import DictConfig
from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config, generate_run_id


@hydra.main(version_base=None, config_path="settings", config_name="main_fat_tree_mip")
def run(config: DictConfig):
    """Run VNE simulation with Fat-Tree topology."""
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")
    print("Running MIP solver on Fat-Tree datacenter topology")
    print(f"Seed: {config.experiment.seed}")
    print(f"Solver: {config.solver.solver_name}")

    if config.experiment.run_id == 'auto':
        config.experiment.run_id = generate_run_id()
    add_simulation_into_config(config)

    system = BaseSystem.from_config(config)
    system.run()

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    run()
