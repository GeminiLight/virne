import numpy as np

from .network import Network


class VirtualNetwork(Network):
    def __init__(self, incoming_graph_data=None, node_attrs_setting=[], link_attrs_setting=[], **kwargs):
        super(VirtualNetwork, self).__init__(incoming_graph_data, node_attrs_setting, link_attrs_setting, **kwargs)

    def generate_topology(self, num_nodes, type='random', **kwargs):
        return super().generate_topology(num_nodes, type=type, **kwargs)

    @property
    def total_node_resource_demand(self):
        n = np.array(self.get_node_attrs_data(self.get_node_attrs('resource'))).sum()
        return n

    @property
    def total_node_resource_demand(self):
        e = np.array(self.get_link_attrs_data(self.get_link_attrs('resource'))).sum()
        return e

    @property
    def total_resource_demand(self):
        n = np.array(self.get_node_attrs_data(self.get_node_attrs('resource'))).sum()
        e = np.array(self.get_link_attrs_data(self.get_link_attrs('resource'))).sum()
        return n + e

if __name__ == '__main__':
    pass