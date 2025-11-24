"""
Modified main.py for running Simulated Annealing (SA) experiments on fat-tree topology with clean switches.

This script extends the standard main.py to:
1. Generate a fat-tree topology for the physical network
2. Set all switch nodes (layer='switch') to CPU=0 (routing-only)
3. Keep host nodes (layer='host') with normal CPU allocation
4. Run the SA meta-heuristic solver with the specified configuration

SA is a probabilistic meta-heuristic optimization algorithm inspired by
the annealing process in metallurgy.

Usage:
    python main_fat_tree_sa.py p_net_setting=fat_tree_p_net_setting \
                                v_sim_setting=v_sim_setting \
                                solver.solver_name=sa_meta \
                                experiment.seed=0
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config, generate_run_id


def set_switches_to_routing_only(p_net):
    """
    Set CPU=0 for all switch nodes (layer='switch') in the physical network.
    This makes switches routing-only, unable to host virtual nodes.

    Args:
        p_net: PhysicalNetwork instance

    Returns:
        Modified PhysicalNetwork instance
    """
    num_switches = 0
    num_hosts = 0

    for node_id in p_net.nodes:
        layer = p_net.nodes[node_id].get('layer', 'host')
        if layer == 'switch':
            # Set CPU to 0 for switches (routing only)
            p_net.nodes[node_id]['cpu'] = 0
            p_net.nodes[node_id]['max_cpu'] = 0
            num_switches += 1
        else:
            num_hosts += 1

    print(f"\nPhysical network configured:")
    print(f"  - Switches (routing-only): {num_switches} nodes with CPU=0")
    print(f"  - Hosts (compute nodes): {num_hosts} nodes with CPU>0")
    print(f"  - Total nodes: {num_switches + num_hosts}")
    print(f"  - Total links: {p_net.num_links}\n")

    return p_net


@hydra.main(version_base=None, config_path="settings", config_name="main_fat_tree_sa")
def run(config: DictConfig):
    """
    Run the VNE simulation with fat-tree topology and clean switches.
    """
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")
    print("Running Simulated Annealing (SA) solver on fat-tree topology with clean switches (routing-only)")
    print(f"Seed: {config.experiment.seed}")
    print(f"Solver: {config.solver.solver_name}")

    # Configure run ID
    if config.experiment.run_id == 'auto':
        config.experiment.run_id = generate_run_id()
    add_simulation_into_config(config)

    # Create system from config
    system = BaseSystem.from_config(config)

    # Modify physical network to set switches to routing-only
    print("\nSetting switches to routing-only (CPU=0)...")
    system.env.p_net = set_switches_to_routing_only(system.env.p_net)

    # Run the simulation
    system.run()

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    run()
