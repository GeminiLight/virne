import pytest
import networkx as nx
from unittest.mock import patch

from virne.network.topology.topology_generator import TopologyGenerator


class TestTopologyGenerator:
    """Test suite for TopologyGenerator class."""
    
    def test_generate_path_graph(self):
        """Test path graph generation."""
        G = TopologyGenerator.generate('path', 5)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 4
        assert nx.is_connected(G)
        
    def test_generate_star_graph(self):
        """Test star graph generation."""
        G = TopologyGenerator.generate('star', 6)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 6
        assert G.number_of_edges() == 5
        assert nx.is_connected(G)
        
    def test_generate_grid_2d_graph(self):
        """Test 2D grid graph generation."""
        G = TopologyGenerator.generate('grid_2d', num_nodes=1, m=3, n=4)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 12  # 3 * 4 = 12 nodes
        assert nx.is_connected(G)
        
    def test_generate_grid_2d_missing_m(self):
        """Test error when 'm' parameter is missing for grid_2d."""
        with pytest.raises(ValueError, match="'grid_2d' type requires 'm' and 'n' keyword arguments"):
            TopologyGenerator.generate('grid_2d', num_nodes=1, n=4)
            
    def test_generate_grid_2d_missing_n(self):
        """Test error when 'n' parameter is missing for grid_2d."""
        with pytest.raises(ValueError, match="'grid_2d' type requires 'm' and 'n' keyword arguments"):
            TopologyGenerator.generate('grid_2d', num_nodes=1, m=3)
            
    def test_generate_grid_2d_missing_both(self):
        """Test error when both 'm' and 'n' parameters are missing for grid_2d."""
        with pytest.raises(ValueError, match="'grid_2d' type requires 'm' and 'n' keyword arguments"):
            TopologyGenerator.generate('grid_2d', num_nodes=1)
            
    @patch('networkx.waxman_graph')
    @patch('networkx.is_connected')
    def test_generate_waxman_graph(self, mock_is_connected, mock_waxman):
        """Test Waxman graph generation."""
        # Create a mock connected graph
        mock_graph = nx.path_graph(5)
        mock_waxman.return_value = mock_graph
        mock_is_connected.return_value = True
        
        G = TopologyGenerator.generate('waxman', 5, wm_alpha=0.4, wm_beta=0.3)
        
        assert G == mock_graph
        mock_waxman.assert_called_with(5, 0.4, 0.3)
        mock_is_connected.assert_called_with(mock_graph)
        
    @patch('networkx.waxman_graph')
    @patch('networkx.is_connected')
    def test_generate_waxman_graph_default_params(self, mock_is_connected, mock_waxman):
        """Test Waxman graph generation with default parameters."""
        mock_graph = nx.path_graph(5)
        mock_waxman.return_value = mock_graph
        mock_is_connected.return_value = True
        
        G = TopologyGenerator.generate('waxman', 5)
        
        mock_waxman.assert_called_with(5, 0.5, 0.2)  # default alpha=0.5, beta=0.2
        
    @patch('networkx.waxman_graph')
    @patch('networkx.is_connected')
    def test_generate_waxman_graph_retry_until_connected(self, mock_is_connected, mock_waxman):
        """Test that Waxman generation retries until connected."""
        mock_graph = nx.path_graph(5)
        mock_waxman.return_value = mock_graph
        # First two calls return False (not connected), third returns True
        mock_is_connected.side_effect = [False, False, True]
        
        G = TopologyGenerator.generate('waxman', 5)
        
        assert G == mock_graph
        assert mock_waxman.call_count == 3
        assert mock_is_connected.call_count == 3
        
    @patch('networkx.erdos_renyi_graph')
    @patch('networkx.is_connected')
    def test_generate_random_graph(self, mock_is_connected, mock_erdos_renyi):
        """Test random graph generation."""
        mock_graph = nx.path_graph(5)
        mock_erdos_renyi.return_value = mock_graph
        mock_is_connected.return_value = True
        
        G = TopologyGenerator.generate('random', 5, random_prob=0.3)
        
        assert G == mock_graph
        mock_erdos_renyi.assert_called_with(5, 0.3, directed=False)
        mock_is_connected.assert_called_with(mock_graph)
        
    @patch('networkx.erdos_renyi_graph')
    @patch('networkx.is_connected')
    def test_generate_random_graph_default_prob(self, mock_is_connected, mock_erdos_renyi):
        """Test random graph generation with default probability."""
        mock_graph = nx.path_graph(5)
        mock_erdos_renyi.return_value = mock_graph
        mock_is_connected.return_value = True
        
        G = TopologyGenerator.generate('random', 5)
        
        mock_erdos_renyi.assert_called_with(5, 0.5, directed=False)  # default prob=0.5
        
    @patch('networkx.erdos_renyi_graph')
    @patch('networkx.is_connected')
    def test_generate_random_graph_retry_until_connected(self, mock_is_connected, mock_erdos_renyi):
        """Test that random generation retries until connected."""
        mock_graph = nx.path_graph(5)
        mock_erdos_renyi.return_value = mock_graph
        # First call returns False (not connected), second returns True
        mock_is_connected.side_effect = [False, True]
        
        G = TopologyGenerator.generate('random', 5)
        
        assert G == mock_graph
        assert mock_erdos_renyi.call_count == 2
        assert mock_is_connected.call_count == 2
        
    def test_generate_invalid_num_nodes_zero(self):
        """Test error with zero nodes."""
        with pytest.raises(AssertionError, match="num_nodes must be >= 1"):
            TopologyGenerator.generate('path', 0)
            
    def test_generate_invalid_num_nodes_negative(self):
        """Test error with negative nodes."""
        with pytest.raises(AssertionError, match="num_nodes must be >= 1"):
            TopologyGenerator.generate('path', -1)
            
    def test_generate_unsupported_type(self):
        """Test error with unsupported graph type."""
        with pytest.raises(NotImplementedError, match="Graph type 'unsupported' is not implemented"):
            TopologyGenerator.generate('unsupported', 5)
            
    def test_generate_path_single_node(self):
        """Test path graph with single node."""
        G = TopologyGenerator.generate('path', 1)
        
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0
        
    def test_generate_star_single_node(self):
        """Test star graph with single node."""
        G = TopologyGenerator.generate('star', 2)
        
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        
    def test_generate_grid_2d_single_cell(self):
        """Test 2D grid with single cell."""
        G = TopologyGenerator.generate('grid_2d', num_nodes=1, m=1, n=1)
        
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0
        
    def test_topology_generator_is_static(self):
        """Test that TopologyGenerator methods are static."""
        # Should be able to call without instantiation
        G = TopologyGenerator.generate('path', 3)
        assert isinstance(G, nx.Graph)
        
        # Should also be able to instantiate and call
        generator = TopologyGenerator()
        G2 = generator.generate('path', 3)
        assert isinstance(G2, nx.Graph)


if __name__ == '__main__':
    pytest.main([__file__])
