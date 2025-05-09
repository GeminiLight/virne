# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from functools import cached_property
import numpy as np
import logging # Added for logging

from .base_network import BaseNetwork

logger = logging.getLogger(__name__) # Added logger


class VirtualNetwork(BaseNetwork):
    """
    VirtualNetwork class is a subclass of Network class. It represents a virtual network.

    Properties:
        total_node_resource_demand (float): The total resource demand of all nodes.
        total_link_resource_demand (float): The total resource demand of all links.
        total_resource_demand (float): The total resource demand of all nodes and links.

    Methods:
        generate_topology: Generate a virtual network topology.
    """
    def __init__(self, incoming_graph_data=None, node_attrs_setting=None, link_attrs_setting=None, **kwargs):
        # Ensure default empty lists for settings if None is passed
        if node_attrs_setting is None:
            node_attrs_setting = []
        if link_attrs_setting is None:
            link_attrs_setting = []
        super().__init__(incoming_graph_data, node_attrs_setting, link_attrs_setting, **kwargs)

    def generate_topology(self, num_nodes: int, type: str = 'random', **kwargs):
        """
        Generate the virtual network topology.

        Args:
            num_nodes: The number of nodes in the generated graph.
            type: The type of graph to generate. Defaults to 'random'.
            **kwargs: Additional keyword arguments for topology generation.
        """
        # The parent method already handles logging and attribute setting.
        super().generate_topology(num_nodes, type=type, **kwargs)

    @property
    def total_node_resource_demand(self) -> float:
        """Calculates the total resource demand of all nodes in the virtual network."""
        resource_node_attrs = self.get_node_attrs(types=['resource'])
        if not resource_node_attrs:
            logger.warning("No node attributes of type 'resource' found to calculate total demand.")
            return 0.0
        
        # Assuming get_node_attrs_data returns a list of lists/arrays of resource values
        # And each node might have one or more resource attributes.
        # This sums up all values from all resource attributes across all nodes.
        try:
            demand_data = self.get_node_attrs_data(resource_node_attrs)
            total_demand = sum(np.sum(attr_data) for attr_data in demand_data if attr_data) # Sum each attribute's data then sum results
        except Exception as e:
            logger.error(f"Error calculating total_node_resource_demand: {e}")
            total_demand = 0.0
        return float(total_demand)

    @property
    def total_link_resource_demand(self) -> float:
        """Calculates the total resource demand of all links in the virtual network."""
        resource_link_attrs = self.get_link_attrs(types=['resource'])
        if not resource_link_attrs:
            logger.warning("No link attributes of type 'resource' found to calculate total demand.")
            return 0.0

        try:
            demand_data = self.get_link_attrs_data(resource_link_attrs)
            total_demand = sum(np.sum(attr_data) for attr_data in demand_data if attr_data)
        except Exception as e:
            logger.error(f"Error calculating total_link_resource_demand: {e}")
            total_demand = 0.0
        return float(total_demand)

    @cached_property
    def total_resource_demand(self) -> float:
        """
        Gets the total resource demand of all nodes and links in the virtual network.
        This value is cached after the first calculation.
        """
        return self.total_node_resource_demand + self.total_link_resource_demand

if __name__ == '__main__':
    # Example Usage (requires BaseNetwork and Attribute classes to be defined and importable)
    # from virne.data.attribute import NodeResourceAttribute, LinkResourceAttribute

    logging.basicConfig(level=logging.INFO)

    # Define some attribute settings
    node_cpu_setting = {'name': 'cpu', 'type': 'resource', 'distribution': 'uniform', 'low': 10, 'high': 50, 'dtype': int}
    link_bw_setting = {'name': 'bw', 'type': 'resource', 'distribution': 'uniform', 'low': 100, 'high': 500, 'dtype': int}
    
    vn = VirtualNetwork(node_attrs_setting=[node_cpu_setting], link_attrs_setting=[link_bw_setting], name='TestVN')
    vn.generate_topology(num_nodes=5, type='random', random_prob=0.5) # Ensure 'random_prob' is passed if needed by base
    vn.generate_attrs_data() # Generate data for the attributes

    print(f"Virtual Network: {vn}")
    print(f"Nodes: {vn.nodes(data=True)}")
    print(f"Edges: {vn.edges(data=True)}")
    print(f"Total Node Resource Demand (CPU): {vn.total_node_resource_demand}")
    print(f"Total Link Resource Demand (BW): {vn.total_link_resource_demand}")
    print(f"Total Resource Demand: {vn.total_resource_demand}")

    # Test with no resource attributes
    vn_no_res = VirtualNetwork(name='VN_NoResource')
    vn_no_res.generate_topology(num_nodes=3)
    vn_no_res.generate_attrs_data()
    print(f"\nVN with no resource attributes: {vn_no_res}")
    print(f"Total Node Resource Demand: {vn_no_res.total_node_resource_demand}")
    print(f"Total Link Resource Demand: {vn_no_res.total_link_resource_demand}")
    print(f"Total Resource Demand: {vn_no_res.total_resource_demand}")
    pass