import copy
import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
from omegaconf import DictConfig, OmegaConf

from virne.network.virtual_network import VirtualNetwork
from virne.network.base_network import BaseNetwork
from virne.utils.setting import read_setting


class TestVirtualNetwork:
    """Test suite for VirtualNetwork class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basic_config = read_setting('tests/settings/v_sim_setting/default.yaml')
        
    def test_init_empty(self):
        """Test initialization with no parameters."""
        v_net = VirtualNetwork()
        assert isinstance(v_net, BaseNetwork)
        assert isinstance(v_net, nx.Graph)
        
    def test_init_with_config(self):
        """Test initialization with config."""
        v_net = VirtualNetwork(config=self.basic_config)
        assert len(v_net.node_attrs) == 1
        assert len(v_net.link_attrs) == 1
        assert 'cpu' in v_net.node_attrs
        assert 'bw' in v_net.link_attrs
        
    def test_generate_topology_random_default(self):
        """Test random topology generation (default for virtual networks)."""
        v_net = VirtualNetwork(config=self.basic_config)
        v_net.generate_topology(num_nodes=5)  # type='random' is default
        assert v_net.number_of_nodes() == 5
        
    def test_generate_topology_path(self):
        """Test path topology generation."""
        v_net = VirtualNetwork(config=self.basic_config)
        v_net.generate_topology(num_nodes=4, type='path')
        assert v_net.number_of_nodes() == 4
        assert v_net.number_of_edges() == 3
        
    def test_total_node_resource_demand_empty(self):
        """Test total node resource demand with empty network."""
        v_net = VirtualNetwork(config=self.basic_config)
        assert v_net.total_node_resource_demand == 0.0
        
    def test_total_link_resource_demand_empty(self):
        """Test total link resource demand with empty network."""
        v_net = VirtualNetwork(config=self.basic_config)
        assert v_net.total_link_resource_demand == 0.0
        
    def test_total_resource_demand_empty(self):
        """Test total resource demand with empty network."""
        v_net = VirtualNetwork(config=self.basic_config)
        assert v_net.total_resource_demand == 0.0
        
    @patch.object(VirtualNetwork, 'get_node_attrs_data')
    def test_total_node_resource_demand_with_data(self, mock_get_node_attrs_data):
        """Test total node resource demand with data."""
        v_net = VirtualNetwork(config=self.basic_config)
        v_net.generate_topology(num_nodes=3, type='path')
        
        # Mock the resource attribute data
        mock_get_node_attrs_data.return_value = [2.0, 3.0, 4.0]
        
        total_demand = v_net.total_node_resource_demand
        assert total_demand == 9.0
        
    @patch.object(VirtualNetwork, 'get_link_attrs_data')
    def test_total_link_resource_demand_with_data(self, mock_get_link_attrs_data):
        """Test total link resource demand with data."""
        v_net = VirtualNetwork(config=self.basic_config)
        v_net.generate_topology(num_nodes=3, type='path')
        
        # Mock the resource attribute data
        mock_get_link_attrs_data.return_value = [5.0, 6.0]
        
        total_demand = v_net.total_link_resource_demand
        assert total_demand == 11.0
        
    @patch.object(VirtualNetwork, 'get_node_attrs_data')
    @patch.object(VirtualNetwork, 'get_link_attrs_data')
    def test_total_resource_demand_with_data(self, mock_get_link_attrs_data, mock_get_node_attrs_data):
        """Test total resource demand with both node and link data."""
        v_net = VirtualNetwork(config=self.basic_config)
        v_net.generate_topology(num_nodes=3, type='path')
        
        # Mock the resource attribute data
        mock_get_node_attrs_data.return_value = [2.0, 3.0, 4.0]
        mock_get_link_attrs_data.return_value = [5.0, 6.0]
        
        total_demand = v_net.total_resource_demand
        assert total_demand == 20.0  # 9.0 + 11.0
        
    @patch.object(VirtualNetwork, 'get_node_attrs_data')
    def test_total_node_resource_demand_exception_handling(self, mock_get_node_attrs_data):
        """Test exception handling in total_node_resource_demand."""
        v_net = VirtualNetwork(config=self.basic_config)
        
        # Mock to raise an exception
        mock_get_node_attrs_data.side_effect = Exception("Test exception")
        
        total_demand = v_net.total_node_resource_demand
        assert total_demand == 0.0
        
    @patch.object(VirtualNetwork, 'get_link_attrs_data')
    def test_total_link_resource_demand_exception_handling(self, mock_get_link_attrs_data):
        """Test exception handling in total_link_resource_demand."""
        v_net = VirtualNetwork(config=self.basic_config)
        
        # Mock to raise an exception
        mock_get_link_attrs_data.side_effect = Exception("Test exception")
        
        total_demand = v_net.total_link_resource_demand
        assert total_demand == 0.0
        
    @patch.object(VirtualNetwork, 'get_node_attrs_data')
    @patch.object(VirtualNetwork, 'get_link_attrs_data')
    def test_total_resource_demand_exception_handling(self, mock_get_link_attrs_data, mock_get_node_attrs_data):
        """Test exception handling in total_resource_demand."""
        v_net = VirtualNetwork(config=self.basic_config)
        
        # Mock to raise an exception
        mock_get_node_attrs_data.side_effect = Exception("Test exception")
        mock_get_link_attrs_data.side_effect = Exception("Test exception")
        
        total_demand = v_net.total_resource_demand
        assert total_demand == 0.0
        
    def test_virtual_network_attributes(self):
        """Test virtual network specific attributes."""
        v_net = VirtualNetwork(config=self.basic_config)
        
        # Test that these attributes exist (defined in type hints)
        assert hasattr(v_net, '__annotations__')
        assert 'id' in v_net.__annotations__
        assert 'arrival_time' in v_net.__annotations__
        assert 'lifetime' in v_net.__annotations__
        
    @patch('networkx.write_gml')
    @patch.object(VirtualNetwork, '_prepare_gml_graph')
    def test_to_gml(self, mock_prepare_gml_graph, mock_write_gml):
        """Test GML export functionality."""
        v_net = VirtualNetwork(config=self.basic_config)
        v_net.generate_topology(num_nodes=3, type='path')
        
        # Mock the GML preparation
        mock_gml_graph = Mock()
        mock_prepare_gml_graph.return_value = mock_gml_graph
        
        test_path = '/test/path/network.gml'
        v_net.to_gml(test_path)
        
        mock_prepare_gml_graph.assert_called_once()
        mock_write_gml.assert_called_once_with(mock_gml_graph, test_path)

    def test_gml_roundtrip_preserves_typed_metadata(self, tmp_path):
        """Saved settings must retain bool and numeric types after loading."""
        config = copy.deepcopy(self.basic_config)
        config['graph_attrs_setting'] = {
            'id': 7,
            'arrival_time': 1.5,
            'lifetime': 20.0,
        }
        original = VirtualNetwork(config=config)
        original.generate_topology(num_nodes=4, type='path')
        original.generate_attrs_data()
        path = tmp_path / 'typed.gml'

        original.to_gml(path)
        loaded = VirtualNetwork.from_gml(path)

        assert loaded.graph['node_attrs_setting'] == original.graph['node_attrs_setting']
        assert loaded.graph['link_attrs_setting'] == original.graph['link_attrs_setting']
        assert loaded.graph['topology'] == original.graph['topology']
        assert loaded.graph['output'] == original.graph['output']
        assert isinstance(loaded.node_attrs['cpu'].generative, bool)
        assert isinstance(loaded.node_attrs['cpu'].low, int)
        assert isinstance(loaded.node_attrs['cpu'].high, int)
        loaded.generate_attrs_data()

    def test_legacy_gml_metadata_is_type_normalized(self, tmp_path):
        """GML files written before typed metadata existed must remain usable."""
        original = VirtualNetwork(config=copy.deepcopy(self.basic_config))
        original.generate_topology(num_nodes=4, type='path')
        original.generate_attrs_data()
        legacy_graph = original._prepare_gml_graph()
        legacy_graph.graph.pop('virne_metadata_json', None)
        path = tmp_path / 'legacy.gml'
        nx.write_gml(legacy_graph, path)

        loaded = VirtualNetwork.from_gml(path)

        assert isinstance(loaded.node_attrs['cpu'].generative, bool)
        assert isinstance(loaded.node_attrs['cpu'].low, int)
        assert isinstance(loaded.node_attrs['cpu'].high, int)
        assert isinstance(loaded.graph['topology']['random_prob'], float)
        loaded.generate_attrs_data()
        
    def test_inheritance_properties(self):
        """Test that VirtualNetwork properly inherits from BaseNetwork."""
        v_net = VirtualNetwork(config=self.basic_config)
        v_net.generate_topology(num_nodes=4, type='path')
        
        # Test inherited properties
        assert v_net.num_nodes == 4
        assert v_net.num_links == 3
        assert hasattr(v_net, 'node_attrs')
        assert hasattr(v_net, 'link_attrs')
        
    def test_with_dictconfig(self):
        """Test initialization with DictConfig."""
        config = OmegaConf.create(self.basic_config)
        v_net = VirtualNetwork(config=config)
        assert len(v_net.node_attrs) == 1
        assert len(v_net.link_attrs) == 1
        
    @patch.object(VirtualNetwork, 'get_node_attrs')
    def test_total_node_resource_demand_no_resource_attrs(self, mock_get_node_attrs):
        """Test total node resource demand when no resource attributes exist."""
        v_net = VirtualNetwork(config=self.basic_config)
        
        # Mock to return empty list (no resource attributes)
        mock_get_node_attrs.return_value = []
        
        total_demand = v_net.total_node_resource_demand
        assert total_demand == 0.0
        
    @patch.object(VirtualNetwork, 'get_link_attrs')
    def test_total_link_resource_demand_no_resource_attrs(self, mock_get_link_attrs):
        """Test total link resource demand when no resource attributes exist."""
        v_net = VirtualNetwork(config=self.basic_config)
        
        # Mock to return empty list (no resource attributes)
        mock_get_link_attrs.return_value = []
        
        total_demand = v_net.total_link_resource_demand
        assert total_demand == 0.0


if __name__ == '__main__':
    pytest.main([__file__])
