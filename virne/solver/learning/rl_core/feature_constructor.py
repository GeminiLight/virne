import numpy as np
from gym import spaces
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Type

from virne.network import VirtualNetwork, PhysicalNetwork
from virne.network import AttributeBenchmarkManager, AttributeBenchmarks, TopologicalMetrics, TopologicalMetricCalculator
from virne.solver.learning.obs_handler import ObservationHandler
from virne.core import Solution, Controller


class BaseFeatureConstructor:
    """
    Abstract base class for feature construction. Extend this for custom observation logic.
    All required state should be passed as arguments to methods, not stored as instance attributes.
    """
    def __init__(
            self, 
            p_net: PhysicalNetwork,
            v_net: VirtualNetwork,
            config: Optional[DictConfig] = None
        ):
        self.obs_handler = ObservationHandler()
        self.config = config
        self.p_net = p_net
        self.v_net = v_net
        p_net_attribute_benchmarks = AttributeBenchmarkManager.get_from_cache('p_net')
        p_net_topological_metrics = TopologicalMetricCalculator.get_from_cache('p_net')
        v_net_topological_metrics = TopologicalMetricCalculator.get_from_cache('v_net')
        if p_net_attribute_benchmarks is None:
            p_net_attribute_benchmarks = AttributeBenchmarkManager.get_benchmarks(p_net, node_attrs=True, link_attrs=True, link_sum_attrs=True)
        if p_net_topological_metrics is None:
            p_net_topological_metrics = TopologicalMetricCalculator.calculate(p_net, degree=True, closeness=True, eigenvector=True, betweenness=True)
        if v_net_topological_metrics is None:
            v_net_topological_metrics = TopologicalMetricCalculator.calculate(v_net, degree=True, closeness=True, eigenvector=True, betweenness=True)
        self.p_net_attribute_benchmarks = p_net_attribute_benchmarks
        self.node_attr_benchmarks = p_net_attribute_benchmarks.node_attr_benchmarks
        self.link_attr_benchmarks = p_net_attribute_benchmarks.link_attr_benchmarks
        self.link_sum_attr_benchmarks = p_net_attribute_benchmarks.link_sum_attr_benchmarks

        self.p_net_topological_metrics = p_net_topological_metrics
        self.v_net_topological_metrics = v_net_topological_metrics
        self.extracted_attr_types = config.rl.feature_constructor.extracted_attr_types
        self.if_use_degree_metric = config.rl.feature_constructor.if_use_degree_metric
        self.if_use_more_topological_metrics = config.rl.feature_constructor.if_use_more_topological_metrics
        self.if_use_aggregated_link_attrs = config.rl.feature_constructor.if_use_aggregated_link_attrs
        self.if_use_node_status_flags = config.rl.feature_constructor.if_use_node_status_flags

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[str, Any]:
        raise NotImplementedError

    def guess_observation_space(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Optional[Solution] = None, v_node_id: int = 0) -> Dict[str, Any]:
        """
        This method is used to guess the observation space based on the physical and virtual networks.
        It should return a dictionary with keys as observation names and values as their respective spaces.
        """
        num_p_net_node_attrs = len(p_net.get_node_attrs(self.extracted_attr_types))
        num_p_net_link_attrs = len(p_net.get_link_attrs(self.extracted_attr_types))
        num_obs_attrs = num_p_net_node_attrs + num_p_net_link_attrs + 2
        spaces.Dict({
            'p_net_x': spaces.Box(low=0, high=1, shape=(p_net.num_nodes, num_p_net_node_attrs), dtype=np.float32),
            'p_net_edge_index': spaces.Box(low=0, high=p_net.num_nodes, shape=(2, p_net.num_links), dtype=np.int32),
            'p_net_edge_attr': spaces.Box(low=0, high=p_net.num_nodes, shape=(p_net.num_links, 2), dtype=np.int32),
            'v_node': spaces.Box(low=0, high=100, shape=(int(num_p_net_node_attrs/2) + int(num_p_net_link_attrs/2) + 2, ), dtype=np.float32)
        })
        raise NotImplementedError

    def _construct_p_net_features(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[str, Any]:
        """
        node_resource, average_distance, p_net_degreees, p_net_nodes_states, v_node_features
        """
        # ====== Node Data of Physical Network ======
        # Node Data 1: Node Attributes
        p_node_attrs_data = self.obs_handler.get_node_attrs_obs(p_net, node_attr_types=self.extracted_attr_types, node_attr_benchmarks=self.node_attr_benchmarks)
        # Node Data 2: Node Status
        if self.config.rl.feature_constructor.if_use_node_status_flags:
            p_nodes_status = self.obs_handler.get_p_net_nodes_status(p_net, v_net, solution['node_slots'], curr_v_node_id)
        else:
            p_nodes_status = np.zeros((p_net.num_nodes, 0), dtype=np.float32)
        # Node Data 3: Link Aggregated Attributes
        if self.config.rl.feature_constructor.if_use_aggregated_link_attrs:
            p_node_link_min_data = self.obs_handler.get_link_aggr_attrs_obs(p_net, link_attr_types=self.extracted_attr_types, aggr='min', link_attr_benchmarks=self.link_attr_benchmarks)
            p_node_link_mean_data = self.obs_handler.get_link_aggr_attrs_obs(p_net, link_attr_types=self.extracted_attr_types, aggr='mean', link_sum_attr_benchmarks=self.link_attr_benchmarks)
            p_node_link_max_data = self.obs_handler.get_link_aggr_attrs_obs(p_net, link_attr_types=self.extracted_attr_types, aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
            p_node_link_sum_data = self.obs_handler.get_link_aggr_attrs_obs(p_net, link_attr_types=self.extracted_attr_types, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
            node_link_aggr_attrs_data = np.concatenate((p_node_link_min_data, p_node_link_mean_data, p_node_link_max_data, p_node_link_sum_data), axis=-1)
        else:
            node_link_aggr_attrs_data = np.zeros((p_net.num_nodes, 0), dtype=np.float32)
        # Node Data 4: Topological Metrics
        avg_distance = self.obs_handler.get_average_distance(p_net, solution['node_slots'], normalization=True)
        if self.config.rl.feature_constructor.if_use_degree_metric:
            p_net_degree_metrics = self.obs_handler.get_node_topological_metrics(
                p_net, self.p_net_topological_metrics, degree=True, closeness=False, eigenvector=False, betweenness=False)
        else:
            p_net_degree_metrics = np.zeros((p_net.num_nodes, 0), dtype=np.float32)
        if self.config.rl.feature_constructor.if_use_more_topological_metrics:
            p_net_more_topological_metrics = self.obs_handler.get_node_topological_metrics(
                p_net, self.p_net_topological_metrics, degree=False, closeness=True, eigenvector=True, betweenness=True)
        else:
            p_net_more_topological_metrics = np.zeros((p_net.num_nodes, 0), dtype=np.float32)
        p_net_topological_metrics = np.concatenate((avg_distance, p_net_degree_metrics, p_net_more_topological_metrics), axis=-1)
        # Node Data Merging
        node_data = np.concatenate((p_node_attrs_data, p_nodes_status, node_link_aggr_attrs_data, p_net_topological_metrics), axis=-1)
        # ====== Node Data of Virtual Node ======
        # num_node_attrs = len(p_net.get_node_attrs(self.extracted_attr_types))
        # num_link_attrs = len(p_net.get_link_attrs(self.extracted_attr_types))
        # num_p_net_x = num_node_attrs + 1  # avg distance
        # num_p_net_x += 2 if self.if_use_node_status_flags else 0
        # num_p_net_x += num_link_attrs * 4 if self.if_use_aggregated_link_attrs else 0
        # num_p_net_x += 1 if self.if_use_degree_metric else 0
        # num_p_net_x += 3 if self.if_use_more_topological_metrics else 0

        # Edge Index
        edge_index = self.obs_handler.get_link_index_obs(p_net)
        # Edge Attributes
        link_data = self.obs_handler.get_link_attrs_obs(p_net, link_attr_types=self.extracted_attr_types, link_attr_benchmarks=self.link_attr_benchmarks)
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data
        }
        return p_net_obs

    def _construct_v_node_features(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[str, Any]:
        if curr_v_node_id  >= v_net.num_nodes:
            return {'x': np.array([], dtype=np.float32)}
        # ====== Node Data of Virtual Node ======
        # Node Data 1: Node Attributes
        v_node_demand = self.obs_handler.get_v_node_demand(v_net, curr_v_node_id, node_attr_types=self.extracted_attr_types, node_attr_benchmarks=self.node_attr_benchmarks)
        # Node Data 2: Node Status
        if self.config.rl.feature_constructor.if_use_node_status_flags:
            v_net_status = self.obs_handler.get_v_node_status(v_net, curr_v_node_id, p_net.num_nodes)
        else:
            v_net_status = np.zeros((0, ), dtype=np.float32)
        # Node Data 3: Link Aggregated Attributes
        if self.config.rl.feature_constructor.if_use_aggregated_link_attrs:
            v_node_mean_link_demend = self.obs_handler.get_v_node_aggr_link_demands(v_net, curr_v_node_id, aggr='mean', link_attr_types=self.extracted_attr_types, link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_max_link_demend = self.obs_handler.get_v_node_aggr_link_demands(v_net, curr_v_node_id, aggr='max', link_attr_types=self.extracted_attr_types, link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_min_link_demend = self.obs_handler.get_v_node_aggr_link_demands(v_net, curr_v_node_id, aggr='min', link_attr_types=self.extracted_attr_types, link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_sum_link_demend = self.obs_handler.get_v_node_aggr_link_demands(v_net, curr_v_node_id, aggr='sum', link_attr_types=self.extracted_attr_types, link_attr_benchmarks=self.link_sum_attr_benchmarks)
            v_node_aggr_attrs_demand = np.concatenate((v_node_mean_link_demend, v_node_max_link_demend, v_node_min_link_demend, v_node_sum_link_demend), axis=-1)
        else:
            v_node_aggr_attrs_demand = np.zeros((0, ), dtype=np.float32)
        # Node Data 4: Topological Metrics
        num_neighbors = len(v_net.adj[curr_v_node_id]) / v_net.num_nodes
        v_num_neighbors = np.array([num_neighbors], dtype=np.float32)
        # Merging Node Data
        v_node_x = np.concatenate([v_node_demand, v_net_status, v_node_aggr_attrs_demand, v_num_neighbors], axis=0)
        # ====== Node Data of Virtual Node ======
        # num_node_attrs = len(v_net.get_node_attrs(self.extracted_attr_types))
        # num_link_attrs = len(v_net.get_link_attrs(self.extracted_attr_types))
        # num_v_node_x = num_node_attrs + 1  # avg distance
        # num_v_node_x += 2 if self.if_use_node_status_flags else 0
        # num_v_node_x += num_link_attrs * 4 if self.if_use_aggregated_link_attrs else 0
        # num_v_node_x += 0 if self.if_use_degree_metric else 0
        # num_v_node_x += 0 if self.if_use_more_topological_metrics else 0
        return {'x': v_node_x}

    def _construct_v_net_features(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[str, Any]:
        # ====== Node Data of Virtual Network ======
        # Node Data 1: Node Attributes
        v_node_attrs_data = self.obs_handler.get_node_attrs_obs(v_net, node_attr_types=self.extracted_attr_types, node_attr_benchmarks=self.node_attr_benchmarks)
        # Node Data 2: Node Status
        if self.config.rl.feature_constructor.if_use_node_status_flags:
            v_node_status = self.obs_handler.get_v_net_nodes_status(v_net, solution['node_slots'], curr_v_node_id, consist_decision=True, neighbor_flags=True)
        else:
            v_node_status = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        # Node Data 3: Link Aggregated Attributes
        if self.config.rl.feature_constructor.if_use_aggregated_link_attrs:
            v_node_link_min_resource = self.obs_handler.get_link_aggr_attrs_obs(v_net, link_attr_types=self.extracted_attr_types, aggr='min', link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(v_net, link_attr_types=self.extracted_attr_types, aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(v_net, link_attr_types=self.extracted_attr_types, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
            v_node_link_mean_resource = self.obs_handler.get_link_aggr_attrs_obs(v_net, link_attr_types=self.extracted_attr_types, aggr='mean', link_sum_attr_benchmarks=self.link_attr_benchmarks)
            v_node_aggr_attrs_data = np.concatenate((v_node_link_min_resource, v_node_link_max_resource, v_node_link_sum_resource, v_node_link_mean_resource), axis=-1)
        else:
            v_node_aggr_attrs_data = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        # Node data 4: topological metrics
        num_neighbors = len(v_net.adj[curr_v_node_id]) / v_net.num_nodes
        v_num_neighbors = np.array([num_neighbors], dtype=np.float32)
        v_num_neighbors = np.expand_dims(v_num_neighbors, axis=1)
        v_num_neighbors = np.ones((v_net.num_nodes, 1), dtype=np.float32) * v_num_neighbors
        if self.config.rl.feature_constructor.if_use_degree_metric:
            v_node_degree_metrics = self.obs_handler.get_node_topological_metrics(
                v_net, self.v_net_topological_metrics, degree=True, closeness=False, eigenvector=False, betweenness=False)
        else:
            v_node_degree_metrics = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        if self.config.rl.feature_constructor.if_use_more_topological_metrics:
            v_node_more_topological_metrics = self.obs_handler.get_node_topological_metrics(
                v_net, self.v_net_topological_metrics, degree=False, closeness=True, eigenvector=True, betweenness=True)
        else:
            v_node_more_topological_metrics = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        v_node_topological_metrics = np.concatenate((v_num_neighbors, v_node_degree_metrics, v_node_more_topological_metrics), axis=-1)
        # Node data 4: topological metrics
        # v_node_neighbor_flags = np.ones((v_net.num_nodes, 1), dtype=np.float32) * self.obs_handler.get_v_node_neighbor_flags(v_net, solution['node_slots'], curr_v_node_id).sum() / 10
        # Merging Node Data
        node_data = np.concatenate((v_node_attrs_data, v_node_status, v_node_aggr_attrs_data, v_node_topological_metrics), axis=-1)
        # ====== Node Data of Virtual Network ======
        # num_node_attrs = len(v_net.get_node_attrs(self.extracted_attr_types))
        # num_link_attrs = len(v_net.get_link_attrs(self.extracted_attr_types))
        # num_v_net_x = num_node_attrs + 1  # avg distance
        # num_v_net_x += 2 if self.if_use_node_status_flags else 0
        # num_v_net_x += num_link_attrs * 4 if self.if_use_aggregated_link_attrs else 0
        # num_v_net_x += 1 if self.if_use_degree_metric else 0
        # num_v_net_x += 3 if self.if_use_more_topological_metrics else 0
        # Edge Index
        edge_index = self.obs_handler.get_link_index_obs(v_net)
        link_data = self.obs_handler.get_link_attrs_obs(v_net, link_attr_types=self.extracted_attr_types, link_attr_benchmarks=self.link_attr_benchmarks)
        v_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data,
        }
        return v_net_obs

class FeatureConstructorRegistry:
    """
    Registry for feature constructor classes. Supports registration and retrieval by name.
    """
    _registry: Dict[str, Type[BaseFeatureConstructor]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(handler_cls: Type[BaseFeatureConstructor]):
            if name in cls._registry:
                raise ValueError(f"Reward calculator '{name}' is already registered.")
            cls._registry[name] = handler_cls
            return handler_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseFeatureConstructor]:
        if name not in cls._registry:
            raise NotImplementedError(f"Feature constructor '{name}' is not implemented.")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[BaseFeatureConstructor]]:
        return dict(cls._registry)


@FeatureConstructorRegistry.register('p_net')
class PNetFeatureConstructor(BaseFeatureConstructor):

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[str, Any]:
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        # Concatenate the observations
        combined_obs = {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
        }
        return combined_obs


@FeatureConstructorRegistry.register('p_net_v_node')
class PNetVNodeFeatureConstructor(BaseFeatureConstructor):

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[str, Any]:
        v_node_obs = self._construct_v_node_features(p_net, v_net, solution, curr_v_node_id)
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        # Concatenate the observations
        combined_obs = {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_node_x': v_node_obs['x']
        }
        return combined_obs

@FeatureConstructorRegistry.register('p_net_v_net')
class PNetVNetFeatureConstructor(BaseFeatureConstructor):

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[str, Any]:
        v_net_obs = self._construct_v_net_features(p_net, v_net, solution, curr_v_node_id)
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        # Concatenate the observations
        combined_obs = {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
        }
        return combined_obs


def get_selected_p_net_nodes(solution):
    """
    Get the selected physical network nodes based on the solution.
    """
    return list(solution['node_slots'].values())


def get_placed_v_net_nodes(solution):
    """
    Get the placed virtual network nodes based on the solution.
    """
    return list(solution['node_slots'].keys())