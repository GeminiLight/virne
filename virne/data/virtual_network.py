# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from functools import cached_property
import numpy as np

from .network import Network


class VirtualNetwork(Network):
    """
    VirtualNetwork class is a subclass of Network class. It represents a virtual network.

    Methods:
        generate_topology: Generate a virtual network topology.
        total_node_resource_demand: Get the total resource demand of all nodes in the virtual network.
        total_link_resource_demand: Get the total resource demand of all links in the virtual network.
        total_resource_demand: Get the total resource demand of all nodes and links in the virtual network.
    """
    def __init__(self, incoming_graph_data=None, node_attrs_setting=[], link_attrs_setting=[], **kwargs):
        super(VirtualNetwork, self).__init__(incoming_graph_data, node_attrs_setting, link_attrs_setting, **kwargs)

    def generate_topology(self, num_nodes, type='random', **kwargs):
        return super().generate_topology(num_nodes, type=type, **kwargs)

    @property
    def total_node_resource_demand(self):
        n = np.array(self.get_node_attrs_data(self.get_node_attrs('resource'))).sum()
        return n

    @property
    def total_link_resource_demand(self):
        e = np.array(self.get_link_attrs_data(self.get_link_attrs('resource'))).sum()
        return e

    @cached_property
    def total_resource_demand(self):
        n = np.array(self.get_node_attrs_data(self.get_node_attrs('resource'))).sum()
        e = np.array(self.get_link_attrs_data(self.get_link_attrs('resource'))).sum()
        return n + e

if __name__ == '__main__':
    pass