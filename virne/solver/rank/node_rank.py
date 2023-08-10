# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import abc
from typing import Union
import numpy as np
import networkx as nx

from virne.data import Network


def rank_nodes(network: Network, method: str = 'order', **kwargs):
    """
    General method for ranking nodes in the network, and store the ranking result in the network object.
    
    Args:
        network (Network): Network object.
        method (str, optional): Node ranking method. Defaults to 'order'.
        **kwargs: Keyword arguments for node ranking method.
    """
    node_rank = node_rank_method_dict.get(method, None)
    if node_rank is None:
        raise NotImplementedError(f'Node ranking method {method} is not implemented.')
    node_ranking = node_rank.rank(network)
    network.node_ranking = node_ranking
    network.ranked_nodes = np.array(list(network.node_ranking.keys()))
    network.node_ranking_values = np.array(list(network.node_ranking.values()))
    return node_ranking


class NodeRank(object):
    """Abstract class for node ranking."""
    def __init__(self, **kwargs):
        __metaclass__ = abc.ABCMeta
        super(NodeRank, self).__init__()
    
    @staticmethod
    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """
        Rank nodes in the network.

        Args:
            network (Network): Network object.
            sort (bool, optional): Sort the ranking result. Defaults to True.

        Returns:
            Union[list, dict]: A list or dict of node ranking result.
        """
        raise NotImplementedError


    def __call__(self, network, sort=True):        
        return self.rank(network, sort=sort)


    @staticmethod
    def to_dict(network: Network, node_rank: list, sort: bool = True) -> dict:
        """
        Convert node ranking result to dict.

        Args:
            network (Network): Network object.
            node_rank (list): Node ranking result.
            sort (bool, optional): Sort the ranking result. Defaults to True.

        Returns:
            dict: A dict of node ranking result.
        """
        assert network.num_nodes == len(node_rank)
        node_rank = {node_id: node_rank[i] for i, node_id in enumerate(network.nodes)}
        if sort:
            node_rank = sorted(node_rank.items(), reverse=True, key=lambda x: x[1])
            node_rank = {i: v for i, v in node_rank}
        return node_rank



class OrderNodeRank(NodeRank):
    """
    Node Ranking Strategy with the default order occurring in the network.
    """
    def __init__(self, **kwargs):
        super(OrderNodeRank, self).__init__(**kwargs)

    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the default order occurring in the network."""
        rank_values = 1 / len(network.nodes)
        node_ranking = {node_id: rank_values for node_id in range(len(network.nodes))}
        return node_ranking


class RandomNodeRank(NodeRank):
    """
    Node Ranking Strategy with random metric.
    """
    def __init__(self, **kwargs):
        super(RandomNodeRank, self).__init__(**kwargs)

    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the random strategy."""
        random_node = [n for n in network.nodes]
        np.random.shuffle(random_node)
        return self.to_dict(network, random_node, sort=sort)


class FFDNodeRank(NodeRank):
    """
    Node Ranking Strategy with First Fit Decreasing (FFD) metric.
    """
    def __init__(self, **kwargs):
        super(FFDNodeRank, self).__init__(**kwargs)
    
    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the quantity of node resources."""
        nodes_data = network.get_node_attrs_data(network.get_node_attrs('resource'))
        node_rank = np.array(nodes_data).sum(axis=0)
        return self.to_dict(network, node_rank, sort=sort)


class NRMNodeRank(NodeRank):
    """
    Node Ranking Strategy with Network Resource Metric (NRM).

    References:
        - Zhang et al. "Toward Profit-Seeking Virtual Network Embedding solver via Global ResVirtual Network \
            Embedding Based on Computing, Network, and Storage Resource Constraintsource Capacity". IoTJ, 2018. 
    """
    def __init__(self, **kwargs):
        super(NRMNodeRank, self).__init__(**kwargs)

    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the Network Resource Metric (NRM) of node."""
        free_nodes_data = network.get_node_attrs_data(network.get_node_attrs('resource'))
        free_nodes_data = np.array(free_nodes_data).sum(axis=0)
        free_links_data = np.array(network.get_aggregation_attrs_data(network.get_link_attrs('resource'), aggr='sum', normalized=False))
        free_links_data = free_links_data.sum(axis=0)
        node_rank = free_nodes_data * free_links_data
        return self.to_dict(network, node_rank, sort=sort)


class DegreeWeightedResoureNodeRank(NodeRank):
    """
    Node Ranking Strategy with Degree and Resource (DR) metric.

    References:
        - Zhang et al. "Node Essentiality Assessment and Distributed Collaborative Virtual Network Embedding in Datacenters". TPDS, 2023. 
    """
    def __init__(self, **kwargs):
        super(DegreeWeightedResoureNodeRank, self).__init__(**kwargs)

    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the Degree and Resource (DR) of node."""
        node_degree_dict = dict(network.degree())
        node_degrees = np.array([node_degree_dict[node_id] for node_id in network.nodes])
        free_nodes_data = network.get_node_attrs_data(network.get_node_attrs('resource'))
        free_nodes_data = np.array(free_nodes_data).sum(axis=0)
        node_rank = node_degrees * free_nodes_data
        return self.to_dict(network, node_rank, sort=sort)


class GRCNodeRank(NodeRank):
    """
    Node Ranking Strategy with Global Resource Capacity (GRC) metric.

    References:
        - Gong et al. "Toward Profit-Seeking Virtual Network Embedding solver via Global Resource Capacity". In INFOCOM, 2014.
    """
    def __init__(self, sigma=0.00001, d=0.85, **kwargs):
        super(GRCNodeRank, self).__init__(**kwargs)
        self.sigma = sigma
        self.d = d

    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the Global Resource Capacity (GRC) metric of node."""
        def calc_grc_c(network):
            free_nodes_data = network.get_node_attrs_data(network.get_node_attrs(['resource']))
            sum_nodes_data = np.array(free_nodes_data).sum(axis=0)
            return sum_nodes_data / sum_nodes_data.sum(axis=0)

        def calc_grc_M(network):
            M = network.get_adjacency_attrs_data(network.get_link_attrs(['resource']), normalized=True)
            M = sum(M) / len(M)
            return M

        c = calc_grc_c(network)
        M = calc_grc_M(network)
        c = np.expand_dims(c, axis=0)
        node_rank = c
        delta = np.inf
        while(delta >= self.sigma):
            new_node_rank = (1 - self.d) * c + self.d * node_rank @ M 
            delta = np.linalg.norm(new_node_rank - node_rank)
            node_rank = new_node_rank
        node_rank = np.asarray(node_rank).flatten()
        return self.to_dict(network, node_rank, sort=sort)


class RWNodeRank(NodeRank):
    """
    Node Ranking Strategy with Random Walk (RW) metric.

    References:
        - Cheng et al. "Virtual Network Embedding Through Topology-Aware Node Ranking". In SIGCOMM, 2011.
    """
    def __init__(self, sigma=0.0001, p_J_u=0.15, p_F_u=0.85, **kwargs):
        super(RWNodeRank, self).__init__(**kwargs)
        self.sigma = sigma
        self.p_J_u = p_J_u
        self.p_F_u = p_F_u

    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the Random Walk (RW) metric of node."""
        def normalize_sparse(coo_matrix):
            data_rows = coo_matrix.row
            for id in np.unique(data_rows):
                data_id = np.where(data_rows==id)[0]
                abs_sum = np.sum(np.abs(coo_matrix.data[data_id]))
                if abs_sum != 0:
                    coo_matrix.data[data_id] = coo_matrix.data[data_id] / abs_sum

        def cal_h_u(network):
            free_nodes_data = network.get_node_attrs_data(network.get_node_attrs('resource'))
            free_nodes_data = np.array(free_nodes_data).sum(axis=0)
            M = network.get_adjacency_attrs_data(network.get_link_attrs('resource'))
            M = sum(M) / len(M)
            bw_data = M.sum(axis=0)
            h_u = free_nodes_data * bw_data
            return h_u

        h_u = cal_h_u(network)
        nr = h_u / (h_u.sum() + 1e-9)
        P_J_u_v = np.tile(nr, (network.num_nodes, 1))

        adj_matrix = nx.adjacency_matrix(network).tocoo()
        adj_matrix.data = h_u[adj_matrix.nonzero()[1]]
        normalize_sparse(adj_matrix)
        P_F_u_v = adj_matrix.toarray()
        T_matrix = (P_J_u_v * self.p_J_u + P_F_u_v * self.p_F_u).T
        delta = np.inf
        nr = np.expand_dims(nr, axis=0).T
        while(delta >= self.sigma):
            new_nr = T_matrix @ nr
            delta = np.linalg.norm(new_nr - nr)
            nr = new_nr
        nr = np.squeeze(nr.T, axis=0)
        return self.to_dict(network, nr, sort=sort)
    

class NPSNodeRank(NodeRank):
    """
    Node Ranking Strategy with Node Proximity Sensing (NPS) metric.

    References:
        - Fan et al. "Efficient Virtual Network Embedding of Cloud-Based Data Center Networks into Optical Networks". TPDS, 2021.
    """
    def __init__(self, **kwargs):
        super(NPSNodeRank, self).__init__(**kwargs)

    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the Node Proximity Sensing (NPS) metric of node."""
        free_nodes_data = network.get_node_attrs_data(network.get_node_attrs('resource'))
        free_nodes_data = np.array(free_nodes_data).sum(axis=0)
        free_links_data = np.array(network.get_aggregation_attrs_data(network.get_link_attrs('resource'), aggr='sum', normalized=False))
        free_links_data = free_links_data.sum(axis=0)
        nrm_node_rank = free_nodes_data * free_links_data
        nrm_node_rank = self.to_dict(network, nrm_node_rank, sort=sort)
        num_neighbors_list = [len(network.adj[i]) for i in range(network.num_nodes)]

        v_bfs_root = num_neighbors_list.index(max(num_neighbors_list))
        hop_far_v_bfs_root = nx.single_source_dijkstra_path_length(network, v_bfs_root)
        v_ranked_value_list = []
        for v_node_id, hop_count in hop_far_v_bfs_root.items():
            v_ranked_value_list.append([v_node_id, hop_count, nrm_node_rank[v_node_id]])
        if sort:
            v_ranked_value_list.sort(key=lambda x: (x[1], -x[2]))
            node_rank = {v_rank_values[0]: (v_rank_values[1], v_rank_values[2]) for v_rank_values in v_ranked_value_list}
            return node_rank
        else:
            node_rank = {v_rank_values[0]: (v_rank_values[1], v_rank_values[2]) for v_rank_values in v_ranked_value_list}
            return node_rank


node_rank_method_dict = {
    'order': OrderNodeRank(),
    'random': RandomNodeRank(),
    'ffd': FFDNodeRank(),
    'nrm': NRMNodeRank(),
    'grc': GRCNodeRank(),
    'rw': RWNodeRank(),
    'nea': DegreeWeightedResoureNodeRank(),
    'nps': NPSNodeRank()
}