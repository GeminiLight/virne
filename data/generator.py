from .physical_network import PhysicalNetwork
from .vn_simulator import VNSimulator


class Generator:

    @staticmethod
    def generate_dataset(config, pn=True, vns=True):
        if pn:
            Generator.generate_pn_dataset_from_config(config)
        if vns:
            Generator.generate_vns_dataset_from_config(config)

    @staticmethod
    def generate_pn_dataset_from_config(config):
        r"""generate pn dataset with the configuratore."""
        if not isinstance(config, dict):
            config = vars(config)
        pn_setting = config['pn_setting']
        pn = PhysicalNetwork.from_setting(pn_setting)

        pn_dataset_dir = config['pn_dataset_dir']
        pn.save_dataset(pn_dataset_dir)
        if config.get('verbose', 1):
            print(f'save pn dataset in {pn_dataset_dir}')

        new_pn = PhysicalNetwork.load_dataset(pn_dataset_dir)
        return new_pn

    @staticmethod
    def generate_vns_dataset_from_config(config):
        r"""generate vn dataset with the configuratore."""
        if not isinstance(config, dict):
            config = vars(config)
        vns_setting = config['vns_setting']
        vn_simulator = VNSimulator.from_setting(vns_setting)
        vn_simulator.renew()
        
        vns_dataset_dir = config['vns_dataset_dir']
        vn_simulator.save_dataset(vns_dataset_dir)
        if config.get('verbose', 1):
            print(f'save vn dataset in {vns_dataset_dir}')

        new_vn_simulator = VNSimulator.load_dataset(vns_dataset_dir)
        return new_vn_simulator
        