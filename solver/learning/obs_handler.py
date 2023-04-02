import numpy as np
import networkx as nx

from functools import lru_cache

from data import Network, PhysicalNetwork, VirtualNetwork


def calc_positional_embeddings(max_len, embedding_dim):
    pe = np.zeros(shape=(max_len, embedding_dim), dtype=np.float32)
    position = np.expand_dims(np.arange(0, max_len), axis=1)
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

MAX_NUM_V_NODES = 100
POSITIONAL_EMBEDDING_DIM = 8
POSITIONAL_EMBEDDINGS = calc_positional_embeddings(MAX_NUM_V_NODES, POSITIONAL_EMBEDDING_DIM)


class ObservationHandler:

    def get_node_order_obs(self, network: Network):
        num_nodes = network.num_nodes
        node_order = np.arange(num_nodes, dtype=np.int32)
        return node_order

    def _get_attr_benchmarks(self, attr_types: list, attrs_list: list, attr_data: np.ndarray) -> dict:
        """Get attributes benchmark for normlization

        Args:
            attr_types: An list of specified types of attributes.
            attrs_list: An list of specified attributes.
            attr_data: The data of these attributes.

        Returns:
            attr_benchmarks: A dict like {attr_name: attr_benchmark}
        """
        attr_benchmarks = {}
        if 'extrema' in attr_types:
            for attr, attr_data in zip(attrs_list, attr_data):
                # for resource attributes, the maximum of extrema attributes are used as its benchmark
                if attr.type == 'resource':
                    continue
                elif attr.type == 'extrema':
                    attr_benchmarks[attr.originator] = attr_data.max()
                else:
                    attr_benchmarks[attr.name] = attr_data.max()
        else:
            for attr, attr_data in zip(attrs_list, attr_data):
                attr_benchmarks[attr.name] = attr_data.max()
        return attr_benchmarks

    def get_degree_benchmark(self, network):
        return max(list(dict(network.degree).values()))

    def get_node_attr_benchmarks(self, network: Network, node_attr_types: list = ['resource', 'extrema']):
        n_attrs = network.get_node_attrs(node_attr_types)
        node_data = np.array(network.get_node_attrs_data(n_attrs), dtype=np.float32)
        node_attr_benchmarks = self._get_attr_benchmarks(node_attr_types, n_attrs, node_data)
        return node_attr_benchmarks
    
    def get_link_attr_benchmarks(self, network: Network, link_attr_types=['resource', 'extrema']):
        l_attrs = network.get_link_attrs(link_attr_types)
        link_data = np.array(network.get_link_attrs_data(l_attrs), dtype=np.float32)
        link_data = np.concatenate([link_data, link_data], axis=1)
        link_attr_benchmarks = self._get_attr_benchmarks(link_attr_types, l_attrs, link_data)
        return link_attr_benchmarks

    def get_link_sum_attr_benchmarks(self, network: Network, link_attr_types=['resource', 'extrema']):
        l_attrs = network.get_link_attrs(link_attr_types)
        link_sum_attrs_data = np.array(network.get_aggregation_attrs_data(l_attrs, aggr='sum'), dtype=np.float32)
        link_sum_attr_benchmarks = self._get_attr_benchmarks(link_attr_types, l_attrs, link_sum_attrs_data)
        return link_sum_attr_benchmarks

    def get_node_degree_obs(self, network, degree_benchmark=None):
        network_degree = np.array(list(dict(network.degree).values()), dtype=np.float32).reshape((network.num_nodes, 1))
        if degree_benchmark is not None:
            network_degree /= degree_benchmark
        return network_degree

    def get_node_attrs_obs(self, network: Network, node_attr_types=['resource', 'extrema'], node_attr_benchmarks=None):
        n_attrs = network.get_node_attrs(node_attr_types)
        node_data = np.array(network.get_node_attrs_data(n_attrs), dtype=np.float32)

        if node_attr_benchmarks is not None:
            for i, n_attr in enumerate(n_attrs):
                attr_name = n_attr.originator if n_attr.type == 'extrema' else n_attr.name
                node_data[i] = node_data[i] / node_attr_benchmarks[attr_name]
        return node_data.T

    def get_link_attrs_obs(self, network: Network, link_attr_types=['resource', 'extrema'], link_attr_benchmarks: dict = None):
        l_attrs = network.get_link_attrs(link_attr_types)
        link_data = np.array(network.get_link_attrs_data(l_attrs), dtype=np.float32)
        link_data = np.concatenate([link_data, link_data], axis=1)

        if link_attr_benchmarks is not None:
            for i, l_attr in enumerate(l_attrs):
                attr_name = l_attr.originator if l_attr.type == 'extrema' else l_attr.name
                link_data[i] = link_data[i] / link_attr_benchmarks[attr_name]
        return link_data.T

    def get_link_sum_attrs_obs(self, network: Network, link_attr_types: list = ['resource', 'extrema'], link_sum_attr_benchmarks: dict = None):
        return self.get_link_aggr_attrs_obs(network, link_attr_types, aggr='sum', link_sum_attr_benchmarks=link_sum_attr_benchmarks)

    def get_link_aggr_attrs_obs(self, network: Network, link_attr_types: list = ['resource', 'extrema'], aggr='sum', link_sum_attr_benchmarks: dict = None, link_attr_benchmarks: dict = None):
        l_attrs = network.get_link_attrs(link_attr_types)
        link_aggr_attrs_data = np.array(network.get_aggregation_attrs_data(l_attrs, aggr=aggr), dtype=np.float32)
        if aggr=='sum' and link_sum_attr_benchmarks is not None:
            for i, l_attr in enumerate(l_attrs):
                attr_name = l_attr.originator if l_attr.type == 'extrema' else l_attr.name
                link_aggr_attrs_data[i] = link_aggr_attrs_data[i] / link_sum_attr_benchmarks[attr_name]
        elif aggr in ['mean', 'max', 'max'] and link_attr_benchmarks is not None:
            for i, l_attr in enumerate(l_attrs):
                attr_name = l_attr.originator if l_attr.type == 'extrema' else l_attr.name
                link_aggr_attrs_data[i] = link_aggr_attrs_data[i] / link_attr_benchmarks[attr_name]
        return link_aggr_attrs_data.T

    def get_link_filters(self, network, resource_threshold: dict):
        threshold = np.array(list(resource_threshold.values()))
        l_attrs_data = np.array(network.get_link_attrs_data(list(resource_threshold.keys()))).T
        link_filters = (l_attrs_data >= threshold).all(axis=1)
        link_filters = np.concatenate([link_filters, link_filters], axis=0)
        return link_filters

    def get_subgraph_view(self, network, resource_threshold: dict):
        def available_link(n1, n2):
            for k, v in resource_threshold.items():
                p_link = network.links[(n1, n2)]
            return p_link[k] >= v
        sub_graph = network.get_subgraph_view(filter_edge=available_link)
        return sub_graph

    def get_link_index_obs(self, network: Network):
        """Get the link index of network.
        
        Args:
            network: the given network.

        Returns:
            link_index: its shape is [2, num_links].
        """
        return self.get_link_pair_obs(network).T

    def get_link_pair_obs(self, network: Network):
        """Get the link index of network.

        Args:
            network: the given network.
        
        Returns:
            link_index: its shape is [2, num_links].
        """
        link_pairs = np.array(list(network.links), dtype=np.int64)
        link_pairs = np.concatenate([link_pairs, link_pairs[:, [1,0]]], axis=0)
        return link_pairs

    def get_selected_node_mask(self, network: Network, selected_nodes):
        selected_node_mask = np.zeros(network.num_nodes, dtype=np.float32)
        selected_node_mask[selected_nodes] = 1.
        return np.array([selected_node_mask], dtype=np.float32, normalization=True)

    def get_average_distance(self, network: Network, nodes_slots, normalization=True):
        # avg_dst
        selected_p_nodes = list(nodes_slots.values())
        if len(selected_p_nodes) == 0: 
            avg_distance = np.zeros(network.num_nodes)
        else:
            distance_dict = dict(nx.shortest_path_length(network))
            avg_distance = []
            for u in range(network.num_nodes):
                sum_dst = 0
                for v in selected_p_nodes:
                    sum_dst += distance_dict[u][v]
                sum_dst /= (len(selected_p_nodes) + 1)
                avg_distance.append(sum_dst)
            avg_distance = np.array(avg_distance)
        if normalization:
            if np.max(avg_distance) == 0:
                 avg_distance = avg_distance
            else:
                avg_distance = (avg_distance - np.min(avg_distance)) / (np.max(avg_distance) - np.min(avg_distance))
        return np.array([avg_distance], dtype=np.float32).T

    def get_average_distance_for_v_node(self, p_net, v_net, nodes_slot, v_node_id=None, normalization=True):
        """Only consider the average distance between each physical node and the neighbors of physical node placed current v nodes
        
        """
        # avg_dst
        if v_node_id is None or len(nodes_slot):
            avg_distance = np.zeros(p_net.num_nodes)
        else:
            distance_dict = dict(nx.shortest_path_length(p_net))
            avg_distance = []
            for p_u in range(p_net.num_nodes):
                sum_dst = 0
                for p_v, v_n in nodes_slot.items():
                    if v_n not in list(v_net.adj[v_node_id]):
                        pass
                    sum_dst += distance_dict[p_u][p_v]
                sum_dst /= (len(p_net) + 1)
                avg_distance.append(sum_dst)
        if normalization:
            if np.max(avg_distance) == 0:
                avg_distance = np.zeros(p_net.num_nodes)
            else:
                avg_distance = (avg_distance - np.min(avg_distance)) / (np.max(avg_distance) - np.min(avg_distance))
        return np.array([avg_distance], dtype=np.float32).T

    def get_v2p_node_link_demand(self, p_net, v_net, node_slots, v_node_id, link_attr_benchmarks=None):
        link_resource_attrs = v_net.get_link_attrs(['resource'])
        v2p_node_link_demands = np.zeros((p_net.num_nodes, len(link_resource_attrs)), dtype=np.float32)
        for v_node_id, p_node_id in node_slots.items():
            if v_node_id in v_net.adj[v_node_id]:
                v_link = (v_node_id, v_node_id)
                for l_attr_id, l_attr in enumerate(link_resource_attrs):
                    v2p_node_link_demands[p_node_id, l_attr_id] = v_net.links[v_link][l_attr.name]
                    if link_attr_benchmarks is not None:
                        v2p_node_link_demands[p_node_id, l_attr_id] /= link_attr_benchmarks[l_attr.name]
        return v2p_node_link_demands

    def get_v_node_link_demands(self, v_net, v_node_id, link_attr_benchmarks=None):
        l_attrs = v_net.get_link_attrs('resource')
        link_demands = []
        for l_attr in l_attrs:
            link_demand = [v_net.links[(n, v_node_id)][l_attr.name] for n in v_net.adj[v_node_id]]
            link_demands.append(link_demand)
        link_demands = np.array(link_demands, dtype=np.float32)
        if link_attr_benchmarks is not None:
            for i, l_attr in enumerate(l_attrs):
                attr_name = l_attr.name
                link_demands[i] = link_demands[i] / link_attr_benchmarks[attr_name]
            return link_demands
        return link_demands

    def get_v_node_min_link_demend(self, v_net, v_node_id, link_attr_benchmarks=None):
        """
        Returns: 
            [{attr_name: min_attr_demand}]
        """
        link_resourcl_attrs = v_net.get_link_attrs('resource')
        v_link_demands = dict(v_net.adj[v_node_id]).values()
        v_node_min_link_demend = {}
        for l_attr_id, l_attr in enumerate(link_resourcl_attrs):
            v_link_attr_demain_list = [v_link_demand.get(l_attr.name, 0.) for v_link_demand in v_link_demands]
            v_node_min_link_demend[l_attr.name] = min(v_link_attr_demain_list)
            if link_attr_benchmarks is not None:
                v_node_min_link_demend[l_attr.name] /= link_attr_benchmarks[l_attr.name]
        return v_node_min_link_demend

    def get_v_node_max_link_demend(self, v_net, v_node_id: int, link_attr_benchmarks=None):
        """
        Returns: 
            [{attr_name: max_attr_demand}]
        """
        link_resourcl_attrs = v_net.get_link_attrs('resource')
        v_link_demands = dict(v_net.adj[v_node_id]).values()
        v_node_max_link_demend = {}
        for l_attr_id, l_attr in enumerate(link_resourcl_attrs):
            v_link_attr_demain_list = [v_link_demand.get(l_attr.name, 0.) for v_link_demand in v_link_demands]
            v_node_max_link_demend[l_attr.name] = max(v_link_attr_demain_list)
            if link_attr_benchmarks is not None:
                v_node_max_link_demend[l_attr.name] /= link_attr_benchmarks[l_attr.name]
        return v_node_max_link_demend

    def get_v_node_features(self, v_net, v_node_id, node_attr_benchmarks=None, link_attr_benchmarks=None):
        if v_node_id >= v_net.num_nodes:
            v_node_id = 0
            return self.get_v_node_features(v_net, v_node_id)
        norm_unplaced = (v_net.num_nodes - (v_node_id + 1)) / v_net.num_nodes
        norm_all_nodes = v_net.num_nodes / self.p_net.num_nodes
        norm_curr_vid = (v_node_id + 1) / self.p_net.num_nodes
        node_demand = []
        for n_attr in v_net.get_node_attrs('resource'):
            one_v_node_demand = v_net.nodes[v_node_id][n_attr.name]
            one_v_node_demand /= node_attr_benchmarks[n_attr.name] if node_attr_benchmarks is not None else None
            node_demand.append(one_v_node_demand)
        norm_node_demand = np.array(node_demand, dtype=np.float32)

        max_link_demand = []
        min_link_demand = []
        mean_link_demand = []
        num_neighbors = len(v_net.adj[v_node_id]) / v_net.num_nodes
        for l_attr in v_net.get_link_attrs('resource'):
            link_demand = [v_net.links[(n, v_node_id)][l_attr.name] for n in v_net.adj[v_node_id]]
            if link_attr_benchmarks is not None:
                max_link_demand.append(max(link_demand) / link_attr_benchmarks[l_attr.name])
                mean_link_demand.append((sum(link_demand) / len(link_demand)) / link_attr_benchmarks[l_attr.name])
                min_link_demand.append(min(link_demand) / link_attr_benchmarks[l_attr.name])
            else:
                max_link_demand.append(max(link_demand))
                mean_link_demand.append((sum(link_demand) / len(link_demand)))
                min_link_demand.append(min(link_demand))
        v_net_obs = np.concatenate([norm_node_demand, max_link_demand, mean_link_demand, [num_neighbors, norm_unplaced, norm_all_nodes, norm_curr_vid]], axis=0)
        return v_net_obs

    def get_meta_obs(self, p_net, v_net, node_slots, link_data):
        link_consumptions = np.zeros_like(link_data)
        meta_edge_index = [[], []]
        meta_edge_attr = []
        placed_v_net_nodes = list(node_slots.keys())
        link_resource_attrs = v_net.get_link_attrs(['resource'])
        for v_link in v_net.links:
            if v_link[0] in placed_v_net_nodes and v_link[1] in placed_v_net_nodes:
                p_node_a, p_node_b = node_slots[v_link[0]], node_slots[v_link[1]]
                if (p_node_a, p_node_b) in p_net.links:
                    p_pair = (p_node_a, p_node_b) if p_node_a <= p_node_b else (p_node_b, p_node_a)
                    link_id_1 = list(p_net.links).index(p_pair)
                    link_id_2 = link_id_1 + p_net.num_links
                    for l_attr_id, l_attr in enumerate(link_resource_attrs):
                        link_consumptions[link_id_1, l_attr_id] = v_net.links[v_link][l_attr.name]
                        link_consumptions[link_id_2, l_attr_id] = v_net.links[v_link][l_attr.name]
                else:
                    meta_edge_index[0] += [p_node_a, p_node_b]
                    meta_edge_index[1] += [p_node_b, p_node_a]
                    meta_resource_info = [0 for l_attr_id, l_attr in enumerate(link_resource_attrs)]
                    meta_demand_info = [v_net.edges[v_link][l_attr.name] for l_attr_id, l_attr in enumerate(link_resource_attrs)]
                    meta_edge_attr.append(meta_resource_info + meta_demand_info)
                    meta_edge_attr.append(meta_resource_info + meta_demand_info)
        return meta_edge_index, meta_edge_attr, link_consumptions

    def get_p_nodes_status(self, p_net, v_net, node_slots, v_node_id):
        """Get the node status of each physical nodes, including selection flags and neighbor flags.
        
        Returns:
            p_nodes_status: with shape [num_p_nodes, 2]
        """
        p_nodes_status = np.zeros((p_net.num_nodes, 2), dtype=np.float32)
        # set the selection flags of selected p nodes to 1
        selected_p_nodes = list(node_slots.values())
        p_nodes_status[selected_p_nodes, 0] = 1.
        # set the neighbor flags of corresponding p neighbors to 1
        placed_v_nodes = list(node_slots.keys())
        placed_p_neighbors = []
        for v_neighbor in list(v_net.adj[v_node_id].keys()):
            if v_neighbor in placed_v_nodes:
                placed_p_neighbors.append(node_slots[v_neighbor])
        p_nodes_status[placed_p_neighbors, 1] = 1.
        return p_nodes_status



    def get_v_nodes_status(self, v_net, node_slots, v_node_id, consist_decision=True):
        """Get the node status of each virtual nodes, including placement flags and decision flags.
        
        Returns:
            v_nodes_status: with shape [num_v_nodes, 2]
        """
        v_nodes_status = np.zeros((v_net.num_nodes, 2), dtype=np.float32)
        placed_v_nodes = list(node_slots.keys())
        # set the placement flags of placed v nodes to 1
        v_nodes_status[placed_v_nodes, 0] = 1.
        # set the decision flag of the current decided v node to 1
        if v_node_id < v_net.num_nodes:
            if consist_decision:
                v_nodes_status[v_node_id, 1] = 1.
            else:
                v_nodes_status[v_node_id, 1] = -1.
        return v_nodes_status

    def get_v_node_positions(self, v_net_rank_nodes):
        v_node_positions = POSITIONAL_EMBEDDINGS[v_net_rank_nodes]
        return v_node_positions

    def get_p_node_positions(self, p_net, node_slots):
        p_node_positions = np.zeros(shape=(p_net.num_nodes, POSITIONAL_EMBEDDING_DIM), dtype=np.float32)
        for v_node_id, p_node_id in node_slots.items():
            p_node_positions[p_node_id] = POSITIONAL_EMBEDDINGS[v_node_id]
        return p_node_positions