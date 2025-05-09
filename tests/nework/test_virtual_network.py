import pytest
import numpy as np
import networkx as nx
import logging
from unittest.mock import PropertyMock, patch
from virne.data.network.virtual_network import VirtualNetwork


@pytest.fixture
def basic_vn():
    """A VirtualNetwork instance with default initialization."""
    return VirtualNetwork()

@pytest.fixture
def vn_with_resource_settings():
    """A VirtualNetwork instance with predefined resource attribute settings (with required fields)."""
    node_settings = [
        {'name': 'cpu', 'type': 'resource', 'owner': 'node', 'distribution': 'uniform', 'low': 10, 'high': 50, 'dtype': int},
        {'name': 'ram', 'type': 'resource', 'owner': 'node', 'distribution': 'uniform', 'low': 100, 'high': 200, 'dtype': int}
    ]
    link_settings = [
        {'name': 'bw', 'type': 'resource', 'owner': 'link', 'distribution': 'uniform', 'low': 50, 'high': 100, 'dtype': int},
        {'name': 'latency', 'type': 'resource', 'owner': 'link', 'distribution': 'uniform', 'low': 5, 'high': 10, 'dtype': int}
    ]
    vn = VirtualNetwork(node_attrs_setting=node_settings, link_attrs_setting=link_settings)
    return vn

class TestVirtualNetwork:

    def test_initialization_default(self, basic_vn):
        assert isinstance(basic_vn, VirtualNetwork)
        assert basic_vn.num_nodes == 0
        assert basic_vn.graph.get('node_attrs_setting') == []
        assert basic_vn.graph.get('link_attrs_setting') == []
        assert basic_vn.node_attrs == {}
        assert basic_vn.link_attrs == {}

    def test_initialization_with_settings(self):
        node_setting = [{'name': 'cpu', 'type': 'resource', 'owner': 'node', 'distribution': 'uniform', 'low': 10, 'high': 50, 'dtype': int}]
        link_setting = [{'name': 'bw', 'type': 'resource', 'owner': 'link', 'distribution': 'uniform', 'low': 50, 'high': 100, 'dtype': int}]
        vn = VirtualNetwork(node_attrs_setting=node_setting, link_attrs_setting=link_setting, custom_graph_attr="test_val")
        
        assert isinstance(vn, VirtualNetwork)
        assert len(vn.graph['node_attrs_setting']) == 1
        assert vn.graph['node_attrs_setting'][0]['name'] == 'cpu'
        assert len(vn.graph['link_attrs_setting']) == 1
        assert vn.graph['link_attrs_setting'][0]['name'] == 'bw'
        assert 'cpu' in vn.node_attrs
        assert 'bw' in vn.link_attrs
        assert vn.graph['custom_graph_attr'] == "test_val"
        # assert vn.custom_graph_attr== "test_val"


    def test_generate_topology(self, basic_vn):
        basic_vn.generate_topology(num_nodes=5, type='random', random_prob=0.5)
        assert basic_vn.num_nodes == 5
        if 'type' in basic_vn.graph:
            assert basic_vn.graph['type'] == 'random'
        
        basic_vn.generate_topology(num_nodes=3, type='path')
        assert basic_vn.num_nodes == 3
        assert basic_vn.graph.get('type') == 'path'
        assert basic_vn.number_of_edges() == 2

    def test_total_node_resource_demand_no_resource_attrs(self, basic_vn, caplog):
        caplog.set_level(logging.WARNING)
        basic_vn.generate_topology(num_nodes=3)
        assert basic_vn.total_node_resource_demand == 0.0
        assert "No node attributes of type 'resource' found" in caplog.text

    def test_total_node_resource_demand_with_data(self, vn_with_resource_settings):
        vn = vn_with_resource_settings
        vn.generate_topology(num_nodes=3, type='path') # nodes 0, 1, 2
        
        nx.set_node_attributes(vn, {0: 10, 1: 20, 2: 30}, 'cpu')
        nx.set_node_attributes(vn, {0: 100, 1: 150, 2: 200}, 'ram')
        
        expected_demand = (10 + 20 + 30) + (100 + 150 + 200)
        assert vn.total_node_resource_demand == float(expected_demand)

    def test_total_node_resource_demand_empty_graph(self, vn_with_resource_settings):
        assert vn_with_resource_settings.total_node_resource_demand == 0.0

    def test_total_link_resource_demand_no_resource_attrs(self, basic_vn, caplog):
        caplog.set_level(logging.WARNING)
        basic_vn.generate_topology(num_nodes=3, type='path')
        assert basic_vn.total_link_resource_demand == 0.0
        assert "No link attributes of type 'resource' found" in caplog.text

    def test_total_link_resource_demand_with_data(self, vn_with_resource_settings):
        vn = vn_with_resource_settings
        vn.generate_topology(num_nodes=3, type='path') # edges (0,1), (1,2)
        
        nx.set_edge_attributes(vn, {(0,1): 50, (1,2): 70}, 'bw')
        nx.set_edge_attributes(vn, {(0,1): 5, (1,2): 7}, 'latency')

        expected_demand = (50 + 70) + (5 + 7)
        assert vn.total_link_resource_demand == float(expected_demand)

    def test_total_link_resource_demand_empty_graph(self, vn_with_resource_settings):
        assert vn_with_resource_settings.total_link_resource_demand == 0.0
        
    def test_total_link_resource_demand_no_links_in_graph(self, vn_with_resource_settings):
        vn = vn_with_resource_settings
        vn.add_node(0) # Graph with one node, no links
        assert vn.total_link_resource_demand == 0.0

    def test_total_resource_demand(self, vn_with_resource_settings):
        vn = vn_with_resource_settings
        vn.generate_topology(num_nodes=2, type='path') # node 0, 1; edge (0,1)

        nx.set_node_attributes(vn, {0: 10, 1: 20}, 'cpu')
        nx.set_node_attributes(vn, {0: 100, 1: 150}, 'ram')
        nx.set_edge_attributes(vn, {(0,1): 50}, 'bw')
        nx.set_edge_attributes(vn, {(0,1): 5}, 'latency')

        expected_node_demand = (10 + 20) + (100 + 150)
        expected_link_demand = 50 + 5
        expected_total_demand = expected_node_demand + expected_link_demand
        
        assert vn.total_resource_demand == float(expected_total_demand)

    def test_total_resource_demand_is_cached(self):
        # Patch class properties before instance creation for PropertyMock to work as expected
        with patch.object(VirtualNetwork, 'total_node_resource_demand', new_callable=PropertyMock, return_value=10.0) as mock_node, \
             patch.object(VirtualNetwork, 'total_link_resource_demand', new_callable=PropertyMock, return_value=5.0) as mock_link:

            fresh_vn = VirtualNetwork() # Create instance after patching

            # First access: computes and caches
            assert fresh_vn.total_resource_demand == 15.0
            mock_node.assert_called_once()
            mock_link.assert_called_once()

            # Second access: should use cached value
            assert fresh_vn.total_resource_demand == 15.0
            mock_node.assert_called_once() # Call counts should remain 1
            mock_link.assert_called_once()

            # Clear cache and re-verify
            if 'total_resource_demand' in fresh_vn.__dict__:
                del fresh_vn.__dict__['total_resource_demand']
            
            assert fresh_vn.total_resource_demand == 15.0
            assert mock_node.call_count == 2
            assert mock_link.call_count == 2

    def test_total_node_resource_demand_error_in_summation(self, vn_with_resource_settings, caplog, monkeypatch):
        caplog.set_level(logging.ERROR)
        vn = vn_with_resource_settings
        vn.generate_topology(num_nodes=1)
        nx.set_node_attributes(vn, {0: 10}, 'cpu') # Ensure resource_node_attrs is not empty

        def mock_get_node_attrs_data_error(attrs):
            raise ValueError("Test error in node summation")
        
        monkeypatch.setattr(vn, 'get_node_attrs_data', mock_get_node_attrs_data_error)
        
        assert vn.total_node_resource_demand == 0.0
        assert "Error calculating total_node_resource_demand: Test error in node summation" in caplog.text

    def test_total_link_resource_demand_error_in_summation(self, vn_with_resource_settings, caplog, monkeypatch):
        caplog.set_level(logging.ERROR)
        vn = vn_with_resource_settings
        vn.generate_topology(num_nodes=2, type='path') # Edge (0,1)
        nx.set_edge_attributes(vn, {(0,1): 50}, 'bw') # Ensure resource_link_attrs is not empty

        def mock_get_link_attrs_data_error(attrs):
            raise ValueError("Test error in link summation")

        monkeypatch.setattr(vn, 'get_link_attrs_data', mock_get_link_attrs_data_error)

        assert vn.total_link_resource_demand == 0.0
        assert "Error calculating total_link_resource_demand: Test error in link summation" in caplog.text