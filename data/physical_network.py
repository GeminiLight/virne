import os
import copy
import networkx as nx

from .network import Network
from .attribute import NodeInfoAttribute, LinkInfoAttribute


class PhysicalNetwork(Network):
    def __init__(self, incoming_graph_data=None, node_attrs_setting=[], link_attrs_setting=[], **kwargs):
        super(PhysicalNetwork, self).__init__(incoming_graph_data, node_attrs_setting, link_attrs_setting, **kwargs)

    def generate_topology(self, num_nodes, type='waxman', **kwargs):
        return super().generate_topology(num_nodes, type=type, **kwargs)

    @staticmethod
    def from_setting(setting):
        setting = copy.deepcopy(setting)
        node_attrs_setting = setting.pop('node_attrs_setting')
        link_attrs_setting = setting.pop('link_attrs_setting')
        # load topology from file
        try:
            if 'file_path' not in setting['topology']:
                raise Exception
            file_path = setting['topology'].get('file_path')
            net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, link_attrs_setting=link_attrs_setting, **setting)
            G = nx.read_gml(file_path, label='id')
            net.__dict__['graph'].update(G.__dict__['graph'])
            net.__dict__['_node'] = G.__dict__['_node']
            net.__dict__['_adj'] = G.__dict__['_adj']
            n_attr_names = net.nodes[list(net.nodes)[0]].keys()
            for n_attr_name in n_attr_names:
                if n_attr_name not in net.node_attrs.keys():
                    net.node_attrs[n_attr_name] = NodeInfoAttribute(n_attr_name)
            l_attr_names = net.links[list(net.links)[0]].keys()
            for l_attr_name in l_attr_names:
                if l_attr_name not in net.link_attrs.keys():
                    net.link_attrs[l_attr_name] = LinkInfoAttribute(l_attr_name)
            print(f'Loaded the topology from {file_path}')
        except Exception as e:
            num_nodes = setting.pop('num_nodes')
            net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, link_attrs_setting=link_attrs_setting, **setting)
            topology_setting = setting.pop('topology')
            # topology_type = topology_setting.pop('type')
            net.generate_topology(num_nodes, **topology_setting)
        net.generate_attrs_data()
        return net

    @staticmethod
    def from_topology_zoo_setting(topology_zoo_setting):
        setting = copy.deepcopy(topology_zoo_setting)
        node_attrs_setting = setting.pop('node_attrs_setting')
        link_attrs_setting = setting.pop('link_attrs_setting')
        file_path = setting.pop('file_path')
        net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, link_attrs_setting=link_attrs_setting, **setting)
        G = nx.read_gml(file_path, label='id')
        net.__dict__['graph'].update(G.__dict__['graph'])
        net.__dict__['_node'] = G.__dict__['_node']
        net.__dict__['_adj'] = G.__dict__['_adj']
        n_attr_names = net.nodes[list(net.nodes)[0]].keys()
        for n_attr_name in n_attr_names:
            if n_attr_name not in net.node_attrs.keys():
                net.node_attrs[n_attr_name] = NodeInfoAttribute(n_attr_name)
        l_attr_names = net.links[list(net.links)[0]].keys()
        for l_attr_name in l_attr_names:
            if l_attr_name not in net.link_attrs.keys():
                net.link_attrs[l_attr_name] = LinkInfoAttribute(l_attr_name)
        net.generate_attrs_data()
        return net

    def save_dataset(self, dataset_dir):
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        file_path = os.path.join(dataset_dir, 'p_net.gml')
        self.to_gml(file_path)

    @staticmethod
    def load_dataset(dataset_dir):
        if not os.path.exists(dataset_dir):
            raise ValueError(f'Find no dataset in {dataset_dir}.\nPlease firstly generating it.')
        file_path = os.path.join(dataset_dir, 'p_net.gml')
        return PhysicalNetwork.from_gml(file_path)


if __name__ == '__main__':
    pass
