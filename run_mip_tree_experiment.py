"""
Experiment runner for MIP solver on tree topology with clean switches.

This script runs the MIP solver 200 times (200 VNR requests) with 5 different seeds
on a tree topology where switches have CPU=0 (routing-only) and hosts have normal CPU.

Usage:
    python run_mip_tree_experiment.py
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import networkx as nx
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from virne.network.physical_network import PhysicalNetwork
from virne.network.virtual_network_request_simulator import VirtualNetworkRequestSimulator
from virne.system import OnlineSystem
from virne.solver.exact.mip import MipSolver
from virne.core.recorder import Recorder
from virne.core.counter import Counter
from virne.core.logger import Logger
from virne.core.controller import Controller


def set_switches_to_routing_only(p_net: PhysicalNetwork):
    """
    Set CPU=0 for all switch nodes (layer='switch') in the physical network.
    This makes switches routing-only, unable to host virtual nodes.

    Args:
        p_net: PhysicalNetwork instance
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

    print(f"Physical network configured:")
    print(f"  - Switches (routing-only): {num_switches} nodes with CPU=0")
    print(f"  - Hosts (compute nodes): {num_hosts} nodes with CPU>0")
    print(f"  - Total nodes: {num_switches + num_hosts}")
    print(f"  - Total links: {p_net.num_links}")

    return p_net


def run_experiment_with_seed(seed: int, config: DictConfig):
    """
    Run a single experiment with the given seed.

    Args:
        seed: Random seed for reproducibility
        config: Hydra configuration
    """
    print(f"\n{'='*80}")
    print(f"Running experiment with seed {seed}")
    print(f"{'='*80}\n")

    # Update config with current seed
    config.experiment.seed = seed

    # Create physical network with tree topology
    print("Creating physical network with tree topology...")
    p_net = PhysicalNetwork.from_setting(config.p_net_setting)
    p_net.generate_topology()
    p_net.generate_attrs_data()

    # Set switches to routing-only (CPU=0)
    p_net = set_switches_to_routing_only(p_net)

    # Create virtual network simulator
    print(f"\nGenerating {config.v_sim_setting.num_v_nets} virtual network requests...")
    v_net_simulator = VirtualNetworkRequestSimulator.from_setting(
        config.v_sim_setting,
        seed=seed
    )

    # Create system components
    controller = Controller(p_net)
    recorder = Recorder(config.recorder, config.experiment, config.solver)
    counter = Counter(p_net, v_net_simulator)
    logger = Logger(config.logger, config.experiment, config.solver)

    # Create MIP solver
    print("\nInitializing MIP solver...")
    solver = MipSolver(
        controller=controller,
        recorder=recorder,
        counter=counter,
        logger=logger,
        config=config.solver
    )

    # Create and run system
    print(f"\nRunning {config.v_sim_setting.num_v_nets} VNR allocations with MIP solver...")
    print(f"This may take a while...\n")

    system = OnlineSystem(
        p_net=p_net,
        v_net_simulator=v_net_simulator,
        controller=controller,
        recorder=recorder,
        counter=counter,
        logger=logger,
        solver=solver,
        config=config
    )

    # Run the simulation
    system.run()

    # Get results
    summary = counter.summary_records()

    print(f"\n{'='*80}")
    print(f"Results for seed {seed}:")
    print(f"{'='*80}")
    print(f"Acceptance rate: {summary.get('acceptance_rate', 0):.4f}")
    print(f"Success count: {summary.get('success_count', 0)}")
    print(f"Total requests: {config.v_sim_setting.num_v_nets}")
    print(f"Average R2C ratio: {summary.get('avg_r2c_ratio', 0):.4f}")
    print(f"Long-term R2C ratio: {summary.get('long_term_r2c_ratio', 0):.4f}")
    print(f"Total revenue: {summary.get('total_revenue', 0):.2f}")
    print(f"Total cost: {summary.get('total_cost', 0):.2f}")
    print(f"Place failures: {summary.get('place_failure_count', 0)}")
    print(f"Route failures: {summary.get('route_failure_count', 0)}")
    print(f"Early rejections: {summary.get('early_rejection_count', 0)}")
    print(f"\nResults saved to: {recorder.save_dir}")
    print(f"{'='*80}\n")

    return summary


@hydra.main(version_base=None, config_path="settings", config_name="main_tree_mip")
def main(config: DictConfig):
    """
    Main experiment runner.
    Runs MIP solver with 5 different seeds on tree topology with 200 VNR requests.
    """
    # Override config with our custom settings
    config.p_net_setting = OmegaConf.load("settings/p_net_setting/tree_p_net_setting.yaml")
    config.v_sim_setting = OmegaConf.load("settings/v_sim_setting/v_sim_200_requests.yaml")
    config.solver.solver_name = 'mip'
    config.experiment.num_simulations = 1
    config.recorder.if_save_records = True
    config.recorder.if_temp_save_records = True

    print("\n" + "="*80)
    print("MIP SOLVER EXPERIMENT ON TREE TOPOLOGY WITH CLEAN SWITCHES")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Physical network: Tree topology (16 hosts + switches)")
    print(f"  - Virtual network requests: {config.v_sim_setting.num_v_nets}")
    print(f"  - Solver: MIP (Mixed Integer Programming)")
    print(f"  - Seeds: 5 different random seeds")
    print(f"  - Switch configuration: CPU=0 (routing-only)")
    print(f"  - Host configuration: CPU={config.p_net_setting.node_attrs_setting[0]['low']}-{config.p_net_setting.node_attrs_setting[0]['high']}")
    print("="*80 + "\n")

    # Run experiments with 5 different seeds
    seeds = [0, 1, 2, 3, 4]
    all_results = []

    for seed in seeds:
        result = run_experiment_with_seed(seed, config)
        all_results.append(result)

    # Print aggregate results
    print("\n" + "="*80)
    print("AGGREGATE RESULTS ACROSS ALL 5 SEEDS")
    print("="*80)

    avg_acceptance_rate = sum(r.get('acceptance_rate', 0) for r in all_results) / len(all_results)
    avg_r2c_ratio = sum(r.get('avg_r2c_ratio', 0) for r in all_results) / len(all_results)
    avg_long_term_r2c = sum(r.get('long_term_r2c_ratio', 0) for r in all_results) / len(all_results)
    total_success = sum(r.get('success_count', 0) for r in all_results)

    print(f"\nAverage acceptance rate: {avg_acceptance_rate:.4f}")
    print(f"Average R2C ratio: {avg_r2c_ratio:.4f}")
    print(f"Average long-term R2C ratio: {avg_long_term_r2c:.4f}")
    print(f"Total successful embeddings (all seeds): {total_success}")
    print(f"Total VNR requests (all seeds): {len(seeds) * config.v_sim_setting.num_v_nets}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()