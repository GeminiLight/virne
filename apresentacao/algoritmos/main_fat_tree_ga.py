"""
Main script for running Genetic Algorithm experiments on Fat-Tree datacenter topology.

Fat-Tree topology automatically sets switch nodes to CPU=0 (routing-only) via
PhysicalNetwork._set_switch_resources_to_zero() method.

Usage:
    python main_fat_tree_ga.py experiment.seed=0
"""

import hydra
from omegaconf import DictConfig
from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config, generate_run_id


@hydra.main(version_base=None, config_path="settings", config_name="main_fat_tree_ga")
def run(config: DictConfig):
    """Run VNE simulation with Fat-Tree topology."""
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")
    print("Running Genetic Algorithm solver on Fat-Tree datacenter topology")
    print(f"Seed: {config.experiment.seed}")
    print(f"Solver: {config.solver.solver_name}")
    print(f"Topology: Fat-Tree (k={config.p_net_setting.topology.k})")

    # Configure run ID
    if config.experiment.run_id == 'auto':
        config.experiment.run_id = generate_run_id()
    add_simulation_into_config(config)

    # Create and run system
    system = BaseSystem.from_config(config)

    # Physical network automatically sets switch CPU=0
    print(f"\nPhysical network: {system.env.p_net.num_nodes} total nodes")
    print(f"  - Switches automatically configured with CPU=0 (routing-only)")

    system.run()

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    run()
