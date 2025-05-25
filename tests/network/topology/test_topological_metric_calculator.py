import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch

from virne.network.topology.topological_metric_calculator import (
    TopologicalMetricCalculator, 
    TopologicalMetrics
)
from virne.network.base_network import BaseNetwork


class TestTopologicalMetrics:
    """Test suite for TopologicalMetrics dataclass."""
    
    def test_topological_metrics_initialization(self):
        """Test TopologicalMetrics can be initialized with default None values."""
        metrics = TopologicalMetrics()
        
        assert metrics.node_degree_centrality is None
        assert metrics.node_closeness_centrality is None
        assert metrics.node_eigenvector_centrality is None
        assert metrics.node_betweenness_centrality is None
        
    def test_topological_metrics_initialization_with_values(self):
        """Test TopologicalMetrics can be initialized with values."""
        degree_values = np.array([[0.5], [1.0], [0.3]])
        closeness_values = np.array([[0.6], [0.8], [0.4]])
        
        metrics = TopologicalMetrics(
            node_degree_centrality=degree_values,
            node_closeness_centrality=closeness_values
        )
        
        np.testing.assert_array_equal(metrics.node_degree_centrality, degree_values)
        np.testing.assert_array_equal(metrics.node_closeness_centrality, closeness_values)
        assert metrics.node_eigenvector_centrality is None
        assert metrics.node_betweenness_centrality is None


class TestTopologicalMetricCalculator:
    """Test suite for TopologicalMetricCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock a BaseNetwork with a graph attribute
        self.mock_network = BaseNetwork()
        self.mock_network.generate_topology(num_nodes=3, type='path')

        # Create a real small graph for testing
        self.real_graph = nx.path_graph(3)
    @patch('virne.network.topology.topological_metric_calculator.TopologicalMetricCalculator.calculate')
    def test_init_calls_calculate(self, mock_calculate):
        """Test that initialization calls calculate method."""
        mock_metrics = TopologicalMetrics()
        mock_calculate.return_value = mock_metrics
        
        calculator = TopologicalMetricCalculator(
            self.mock_network,
            degree=True,
            closeness=False,
            eigenvector=False,
            betweenness=False
        )
        
        mock_calculate.assert_called_once_with(
            self.mock_network,
            degree=True,
            closeness=False,
            eigenvector=False,
            betweenness=False
        )
        assert calculator.metrics == mock_metrics
        
    @patch('networkx.degree_centrality')
    def test_calculate_degree_only(self, mock_degree_centrality):
        """Test calculate method with only degree centrality."""
        mock_degree_centrality.return_value = {0: 0.5, 1: 1.0, 2: 0.5}
        
        metrics = TopologicalMetricCalculator.calculate(
            self.real_graph,
            degree=True,
            closeness=False,
            eigenvector=False,
            betweenness=False
        )
        
        assert metrics.node_degree_centrality is not None
        assert metrics.node_closeness_centrality is None
        assert metrics.node_eigenvector_centrality is None
        assert metrics.node_betweenness_centrality is None
        
        mock_degree_centrality.assert_called_once_with(self.real_graph)
        
    @patch('networkx.closeness_centrality')
    @patch('networkx.degree_centrality')
    def test_calculate_multiple_metrics(self, mock_degree_centrality, mock_closeness_centrality):
        """Test calculate method with multiple metrics."""
        mock_degree_centrality.return_value = {0: 0.5, 1: 1.0, 2: 0.5}
        mock_closeness_centrality.return_value = {0: 0.6, 1: 0.8, 2: 0.6}
        
        metrics = TopologicalMetricCalculator.calculate(
            self.real_graph,
            degree=True,
            closeness=True,
            eigenvector=False,
            betweenness=False
        )
        
        assert metrics.node_degree_centrality is not None
        assert metrics.node_closeness_centrality is not None
        assert metrics.node_eigenvector_centrality is None
        assert metrics.node_betweenness_centrality is None
        
    @patch('networkx.eigenvector_centrality')
    def test_calculate_eigenvector_centrality(self, mock_eigenvector_centrality):
        """Test calculate method with eigenvector centrality."""
        mock_eigenvector_centrality.return_value = {0: 0.4, 1: 0.8, 2: 0.4}
        
        metrics = TopologicalMetricCalculator.calculate(
            self.real_graph,
            degree=False,
            closeness=False,
            eigenvector=True,
            betweenness=False
        )
        
        assert metrics.node_eigenvector_centrality is not None
        assert metrics.node_degree_centrality is None
        
    @patch('networkx.betweenness_centrality')
    def test_calculate_betweenness_centrality(self, mock_betweenness_centrality):
        """Test calculate method with betweenness centrality."""
        mock_betweenness_centrality.return_value = {0: 0.0, 1: 1.0, 2: 0.0}
        
        metrics = TopologicalMetricCalculator.calculate(
            self.real_graph,
            degree=False,
            closeness=False,
            eigenvector=False,
            betweenness=True
        )
        
        assert metrics.node_betweenness_centrality is not None
        assert metrics.node_degree_centrality is None
        
    @patch('networkx.degree_centrality')
    def test_calculate_with_normalization_disabled(self, mock_degree_centrality):
        """Test calculate method with normalization disabled."""
        mock_degree_centrality.return_value = {0: 0.5, 1: 1.0, 2: 0.5}
        
        metrics = TopologicalMetricCalculator.calculate(
            self.real_graph,
            degree=True,
            closeness=False,
            eigenvector=False,
            betweenness=False,
            normalize=False
        )
        
        assert metrics.node_degree_centrality is not None
        # Values should not be normalized when normalize=False
        
    @patch('networkx.degree_centrality')
    def test_calculate_invalid_centrality_return_type(self, mock_degree_centrality):
        """Test error when centrality function returns non-dict."""
        mock_degree_centrality.return_value = [0.5, 1.0, 0.5]  # list instead of dict
        
        with pytest.raises(TypeError, match="degree_centrality must return a dict"):
            TopologicalMetricCalculator.calculate(
                self.real_graph,
                degree=True,
                closeness=False,
                eigenvector=False,
                betweenness=False
            )
            
    def test_cache_methods(self):
        """Test cache add and get methods."""
        # Clear cache first
        TopologicalMetricCalculator._cache.clear()
        
        metrics = TopologicalMetrics(node_degree_centrality=np.array([[0.5], [1.0]]))
        cache_key = "test_network"
        
        # Test adding to cache
        TopologicalMetricCalculator.add_to_cache(cache_key, metrics)
        
        # Test retrieving from cache
        cached_metrics = TopologicalMetricCalculator.get_from_cache(cache_key)
        
        assert cached_metrics is not None
        assert cached_metrics == metrics
        np.testing.assert_array_equal(
            cached_metrics.node_degree_centrality, 
            metrics.node_degree_centrality
        )
        
    def test_cache_get_nonexistent_key(self):
        """Test getting from cache with non-existent key."""
        result = TopologicalMetricCalculator.get_from_cache("nonexistent_key")
        assert result is None
        
    def test_cache_is_class_variable(self):
        """Test that cache is shared across instances."""
        # Clear cache first
        TopologicalMetricCalculator._cache.clear()
        
        metrics = TopologicalMetrics()
        cache_key = "shared_test"
        
        # Add through class method
        TopologicalMetricCalculator.add_to_cache(cache_key, metrics)
        
        # Should be accessible through any instance
        calculator1 = TopologicalMetricCalculator(self.mock_network)
        calculator2 = TopologicalMetricCalculator(self.mock_network)
        
        assert TopologicalMetricCalculator.get_from_cache(cache_key) == metrics
        assert calculator1.get_from_cache(cache_key) == metrics
        assert calculator2.get_from_cache(cache_key) == metrics
        
    def test_array_dtype_consistency(self):
        """Test that calculated arrays have consistent dtype."""
        with patch('networkx.degree_centrality') as mock_degree:
            mock_degree.return_value = {0: 0.5, 1: 1.0, 2: 0.5}
            
            metrics = TopologicalMetricCalculator.calculate(
                self.real_graph,
                degree=True,
                closeness=False,
                eigenvector=False,
                betweenness=False
            )
            
            assert metrics.node_degree_centrality.dtype == np.float32
            
    def test_array_shape_consistency(self):
        """Test that calculated arrays have correct shape."""
        with patch('networkx.degree_centrality') as mock_degree:
            mock_degree.return_value = {0: 0.5, 1: 1.0, 2: 0.5}
            
            metrics = TopologicalMetricCalculator.calculate(
                self.real_graph,
                degree=True,
                closeness=False,
                eigenvector=False,
                betweenness=False
            )
            
            # Should be column vector: (num_nodes, 1)
            assert metrics.node_degree_centrality.shape == (3, 1)


if __name__ == '__main__':
    pytest.main([__file__])
