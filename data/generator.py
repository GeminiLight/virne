import random
import numpy as np

from .physical_network import PhysicalNetwork
from .virtual_network_request_simulator import VirtualNetworkRequestSimulator
from utils import get_p_net_dataset_dir_from_setting, get_v_nets_dataset_dir_from_setting


class Generator:

    @staticmethod
    def generate_dataset(config, p_net=True, v_nets=True, save=False):
        physical_network = Generator.generate_p_net_dataset_from_config(config, save=save) if p_net else None
        v_net_simulator = Generator.generate_v_nets_dataset_from_config(config, save=save) if v_nets else None
        return physical_network, v_net_simulator

    @staticmethod
    def generate_p_net_dataset_from_config(config, save=False):
        r"""generate p_net dataset with the configuratore."""
        if not isinstance(config, dict):
            config = vars(config)
        p_net_setting = config['p_net_setting']
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        
        p_net = PhysicalNetwork.from_setting(p_net_setting)

        if save:
            p_net_dataset_dir = get_p_net_dataset_dir_from_setting(p_net_setting)
            p_net.save_dataset(p_net_dataset_dir)
            if config.get('verbose', 1):
                print(f'save p_net dataset in {p_net_dataset_dir}')

        # new_p_net = PhysicalNetwork.load_dataset(p_net_dataset_dir)
        return p_net

    @staticmethod
    def generate_v_nets_dataset_from_config(config, save=False):
        r"""generate v_net dataset with the configuratore."""
        if not isinstance(config, dict):
            config = vars(config)
        v_sim_setting = config['v_sim_setting']
        random.seed(config['seed'])
        np.random.seed(config['seed'])

        v_net_simulator = VirtualNetworkRequestSimulator.from_setting(v_sim_setting)
        v_net_simulator.renew()

        if save:
            v_nets_dataset_dir = get_v_nets_dataset_dir_from_setting(v_sim_setting)
            v_net_simulator.save_dataset(v_nets_dataset_dir)
            if config.get('verbose', 1):
                print(f'save v_net dataset in {v_nets_dataset_dir}')

        # new_v_net_simulator = VirtualNetworkRequestSimulator.load_dataset(v_nets_dataset_dir)
        return v_net_simulator
        