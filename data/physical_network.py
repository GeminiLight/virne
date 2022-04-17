import os
import copy

from .network import Network


class PhysicalNetwork(Network):
    def __init__(self, incoming_graph_data=None, node_attrs_setting=[], edge_attrs_setting=[], **kwargs):
        super(PhysicalNetwork, self).__init__(incoming_graph_data, node_attrs_setting, edge_attrs_setting, **kwargs)

    def generate_topology(self, num_nodes, type='waxman', **kwargs):
        return super().generate_topology(num_nodes, type=type, **kwargs)

    @staticmethod
    def from_setting(setting):
        setting = copy.deepcopy(setting)
        node_attrs_setting = setting.pop('node_attrs_setting')
        edge_attrs_setting = setting.pop('edge_attrs_setting')
        num_nodes = setting.pop('num_nodes')
        topology_setting = setting.pop('topology')
        net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, edge_attrs_setting=edge_attrs_setting, **setting)
        topology_type = topology_setting.pop('type')
        net.generate_topology(num_nodes, topology_type, **topology_setting)
        net.generate_attrs_data()
        return net

    def save_dataset(self, dataset_dir):
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        file_path = os.path.join(dataset_dir, 'pn.gml')
        self.to_gml(file_path)

    @staticmethod
    def load_dataset(dataset_dir):
        if not os.path.exists(dataset_dir):
            raise ValueError(f'Find no dataset in {dataset_dir}.\nPlease firstly generating it.')
        file_path = os.path.join(dataset_dir, 'pn.gml')
        return PhysicalNetwork.from_gml(file_path)


if __name__ == '__main__':
    pass
