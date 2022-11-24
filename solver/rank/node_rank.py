import abc
import numpy as np
import networkx as nx



def rank_nodes(network, method='order', **kwargs):
    node_rank = node_rank_method_dict.get(method, None)
    if node_rank is None:
        raise NotImplementedError
    node_ranking = node_rank.rank(network)
    network.node_ranking = node_ranking
    network.ranked_nodes = np.array(list(network.node_ranking.keys()))
    network.node_ranking_values = np.array(list(network.node_ranking.values()))
    return node_ranking


class NodeRank(object):
    def __init__(self, **kwargs):
        __metaclass__ = abc.ABCMeta
        super(NodeRank, self).__init__()
    
    @abc.abstractmethod
    def rank(self, network, sort=True):
        pass

    def __call__(self, network, sort=True):        
        return self.rank(network, sort=sort)

    @staticmethod
    def to_dict(network, node_rank, sort=True):
        assert network.num_nodes == len(node_rank)
        node_rank = {node_id: node_rank[i] for i, node_id in enumerate(network.nodes)}
        if sort:
            node_rank = sorted(node_rank.items(), reverse=True, key=lambda x: x[1])
            node_rank = {i: v for i, v in node_rank}
        return node_rank


class OrderNodeRank(NodeRank):
    r"""
    Rank nodes with the default order occurring in the network.
    """
    def __init__(self, **kwargs):
        super(OrderNodeRank, self).__init__(**kwargs)

    def rank(self, network, sort=True):
        rank_values = 1 / len(network.nodes)
        node_ranking = {node_id: rank_values for node_id in range(len(network.nodes))}
        return node_ranking


class RandomNodeRank(NodeRank):
    r"""
    Rank nodes with the random strategy.
    """
    def __init__(self, **kwargs):
        super(RandomNodeRank, self).__init__(**kwargs)

    def rank(self, network, sort=True):
        random_node = [n for n in network.nodes]
        np.random.shuffle(random_node)
        return self.to_dict(network, random_node, sort=sort)


class FFDNodeRank(NodeRank):
    r"""
    Rank nodes with the quantity of node resources.
    """
    def __init__(self, **kwargs):
        super(FFDNodeRank, self).__init__(**kwargs)
    
    def rank(self, network, sort=True):
        nodes_data = network.get_node_attrs_data(network.get_node_attrs('resource'))
        node_rank = np.array(nodes_data).sum(axis=0)
        return self.to_dict(network, node_rank, sort=sort)


class NRMNodeRank(NodeRank):
    r"""
    An implementation of NRM solver proposed in
    Zhang et al. "Toward Profit-Seeking Virtual Network Embedding solver via Global ResVirtual Network Embedding Based on Computing, Network, and Storage Resource Constraintsource Capacity". IoTJ, 2018. 
    """
    def __init__(self, **kwargs):
        super(NRMNodeRank, self).__init__(**kwargs)

    def rank(self, network, sort=True):
        free_nodes_data = network.get_node_attrs_data(network.get_node_attrs('resource'))
        free_nodes_data = np.array(free_nodes_data).sum(axis=0)
        free_links_data = np.array(network.get_aggregation_attrs_data(network.get_link_attrs('resource'), aggr='sum', normalized=False))
        free_links_data = free_links_data.sum(axis=0)
        node_rank = free_nodes_data * free_links_data
        return self.to_dict(network, node_rank, sort=sort)


class GRCNodeRank(NodeRank):
    r"""
    An implementation of GRC solver proposed in
    Gong et al. "Toward Profit-Seeking Virtual Network Embedding solver via Global Resource Capacity". In INFOCOM, 2014.
    """
    def __init__(self, sigma=0.00001, d=0.85, **kwargs):
        super(GRCNodeRank, self).__init__(**kwargs)
        self.sigma = sigma
        self.d = d

    def rank(self, network, sort=True):
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
    r"""
    An implementation of NodeRank solver proposed in
    Cheng et al. "Virtual Network Embedding Through Topology-Aware Node Ranking". In SIGCOMM, 2011.
    """
    def __init__(self, sigma=0.0001, p_J_u=0.15, p_F_u=0.85, **kwargs):
        super(RWNodeRank, self).__init__(**kwargs)
        self.sigma = sigma
        self.p_J_u = p_J_u
        self.p_F_u = p_F_u

    def rank(self, network, sort=True):
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
    r"""
    An implementation of Node Proximity Sensing solver proposed in
    Zhang et al. "Toward Profit-Seeking Virtual Network Embedding solver via Global ResVirtual Network Embedding Based on Computing, Network, and Storage Resource Constraintsource Capacity". IoTJ, 2018. 
    """
    def __init__(self, **kwargs):
        super(NRMNodeRank, self).__init__(**kwargs)

    def rank(self, network, sort=True):
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
}


if __name__ == '__main__':
    pass