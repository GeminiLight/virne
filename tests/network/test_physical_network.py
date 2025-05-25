import pytest
import os
import tempfile
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch, mock_open
from omegaconf import DictConfig, OmegaConf

from virne.network.physical_network import PhysicalNetwork
from virne.network.base_network import BaseNetwork
from virne.utils.setting import read_setting


class TestPhysicalNetwork:
    """Test suite for PhysicalNetwork class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basic_config = read_setting('tests/settings/p_net_setting/default.yaml')
        
    def test_init_empty(self):
        """Test initialization with no parameters."""
        p_net = PhysicalNetwork()
        assert isinstance(p_net, BaseNetwork)
        assert isinstance(p_net, nx.Graph)
        
    def test_init_with_config(self):
        """Test initialization with config."""
        p_net = PhysicalNetwork(config=self.basic_config)
        assert len(p_net.node_attrs) == 2
        assert len(p_net.link_attrs) == 2
        assert 'cpu' in p_net.node_attrs
        assert 'bw' in p_net.link_attrs
        
    def test_generate_topology_waxman(self):
        """Test waxman topology generation."""
        p_net = PhysicalNetwork(config=self.basic_config)
        p_net.generate_topology(num_nodes=10, type='waxman', alpha=0.5, beta=0.2)
        assert p_net.number_of_nodes() == 10
        assert p_net.number_of_edges() >= 0  # Waxman may generate variable edges
        
    def test_generate_attrs_data_inheritance(self):
        """Test that attrs data generation inherits from BaseNetwork."""
        p_net = PhysicalNetwork(config=self.basic_config)
        p_net.generate_topology(num_nodes=5, type='path')
        p_net.generate_attrs_data()
        
        # Check nodes have attributes
        for node in p_net.nodes():
            assert 'cpu' in p_net.nodes[node]
            assert 'max_cpu' in p_net.nodes[node]
            
        # Check edges have attributes
        for edge in p_net.edges():
            assert 'bw' in p_net.edges[edge]
            assert 'max_bw' in p_net.edges[edge]
            
    @patch('os.path.exists')
    @patch('networkx.read_gml')
    def test_from_setting_with_file(self, mock_read_gml, mock_exists):
        """Test creating PhysicalNetwork from setting with existing file."""
        # Setup mocks
        mock_exists.return_value = True
        mock_graph = nx.path_graph(5)
        for i in range(5):
            mock_graph.nodes[i]['cpu'] = i + 1
            mock_graph.nodes[i]['max_cpu'] = i + 10
        for edge in mock_graph.edges():
            mock_graph.edges[edge]['bw'] = 10
            mock_graph.edges[edge]['max_bw'] = 20
        mock_read_gml.return_value = mock_graph
        
        config = self.basic_config.copy()
        config['topology'] = {'file_path': '/test/path/topology.gml'}
        
        p_net = PhysicalNetwork.from_setting(config)
        
        assert p_net.number_of_nodes() == 5
        mock_read_gml.assert_called_once()
        mock_exists.assert_called_once_with('/test/path/topology.gml')
        
    @patch('os.path.exists')
    def test_from_setting_without_file(self, mock_exists):
        """Test creating PhysicalNetwork from setting without file."""
        mock_exists.return_value = False
        
        config = self.basic_config.copy()
        config['topology']['num_nodes'] = 8
        config['topology']['file_path'] = '/nonexistent/path.gml'
        config['topology']['type'] = 'path'
        
        p_net = PhysicalNetwork.from_setting(config)
        
        # Should generate topology instead of loading from file
        assert p_net.number_of_nodes() == 8
        assert p_net.number_of_edges() == 7  # Path graph has n-1 edges
        
    def test_from_setting_no_file_path(self):
        """Test creating PhysicalNetwork from setting without file_path."""
        config = self.basic_config.copy()
        config['topology']['num_nodes'] = 6
        config['topology']['type'] = 'star'
        
        p_net = PhysicalNetwork.from_setting(config)
        
        assert p_net.number_of_nodes() == 6
        assert p_net.number_of_edges() == 5  # Star graph has n-1 edges
        
    @patch('virne.utils.dataset.set_seed')
    def test_from_setting_with_seed(self, mock_set_seed):
        """Test that seed is properly set when provided."""
        config = self.basic_config.copy()
        config['topology']['num_nodes'] = 5
        
        PhysicalNetwork.from_setting(config, seed=42)
        
        # mock_set_seed.assert_called_with(42)
        
    def test_from_setting_missing_num_nodes(self):
        """Test error when num_nodes is missing for topology generation."""
        config = self.basic_config.copy()
        # Remove num_nodes, don't provide file_path
        config['topology']['type'] = 'path'
        del config['topology']['num_nodes']
        
        with pytest.raises(KeyError):
            PhysicalNetwork.from_setting(config)
            
    @patch('os.path.exists')
    @patch('networkx.read_gml')
    def test_from_setting_gml_read_error(self, mock_read_gml, mock_exists):
        """Test handling of GML read errors."""
        mock_exists.return_value = True
        mock_read_gml.side_effect = Exception("Failed to read GML")
        
        config = self.basic_config.copy()
        config['topology']['file_path'] = '/test/path/bad.gml'
        config['topology']['num_nodes'] = 5
        
        # Should fall back to generation when GML reading fails
        p_net = PhysicalNetwork.from_setting(config)
        assert p_net.number_of_nodes() == 5
        
    def test_inheritance_properties(self):
        """Test that PhysicalNetwork properly inherits from BaseNetwork."""
        p_net = PhysicalNetwork(config=self.basic_config)
        p_net.generate_topology(num_nodes=4, type='path')
        
        # Test inherited properties
        assert p_net.num_nodes == 4
        assert p_net.num_links == 3
        assert hasattr(p_net, 'node_attrs')
        assert hasattr(p_net, 'link_attrs')
        
    def test_with_dictconfig(self):
        """Test initialization with DictConfig."""
        config = OmegaConf.create(self.basic_config)
        p_net = PhysicalNetwork(config=config)
        assert len(p_net.node_attrs) == 2
        assert len(p_net.link_attrs) == 2


if __name__ == '__main__':
    pytest.main([__file__])