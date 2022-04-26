import numpy as np
import networkx as nx


class ObservationHandler:

    def get_node_order_obs(self, network):
        num_nodes = network.num_nodes
        node_order = np.arange(num_nodes, dtype=np.int32)
        return node_order

    def get_node_attrs_obs(self, network, node_attr_types=['resource', 'extrema'], normalization=True, benchmark=None):
        n_attrs = network.get_node_attrs(node_attr_types)
        node_data = np.array(network.get_node_attrs_data(n_attrs), dtype=np.float32)

        if not normalization:
            return node_data, {}

        if benchmark is not None:
            for i, n_attr in enumerate(n_attrs):
                attr_name = n_attr.originator if n_attr.type == 'extrema' else n_attr.name
                node_data[i] = node_data[i] / benchmark[attr_name]
            return node_data, benchmark

        node_attrs_benchmark = {}
        if 'extrema' in node_attr_types:
            for n_attr, n_attr_data in zip(n_attrs, node_data):
                if n_attr.type == 'extrema':
                    node_attrs_benchmark[n_attr.originator] = n_attr_data.max()
            for i, n_attr in enumerate(n_attrs):
                attr_name = n_attr.originator if n_attr.type == 'extrema' else n_attr.name
                node_data[i] = node_data[i] / node_attrs_benchmark[attr_name]
        else:
            for n_attr, n_attr_data in zip(n_attrs, node_data):
                node_attrs_benchmark[n_attr.name] = n_attr_data.max()
            for i, n_attr in enumerate(n_attrs):
                node_data[i] = node_data[i] / node_attrs_benchmark[n_attr.name]
        return node_data, node_attrs_benchmark

    def get_edge_attrs_obs(self, network, edge_attr_types=['resource', 'extrema'], normalization=True, benchmark=None):
        e_attrs = network.get_edge_attrs(edge_attr_types)
        edge_data = np.array(network.get_edge_attrs_data(e_attrs), dtype=np.float32)
        
        if not normalization:
            return edge_data, {}

        if benchmark is not None:
            for i, e_attr in enumerate(e_attrs):
                attr_name = e_attr.originator if e_attr.type == 'extrema' else e_attr.name
                edge_data[i] = edge_data[i] / benchmark[attr_name]
            return edge_data, benchmark

        edge_attrs_benchmark = {}
        if 'extrema' in edge_attr_types:
            for e_attr, e_attr_data in zip(e_attrs, edge_data):
                if e_attr.type == 'extrema':
                    edge_attrs_benchmark[e_attr.originator] = e_attr_data.max()
            for i, e_attr in enumerate(e_attrs):
                attr_name = e_attr.originator if e_attr.type == 'extrema' else e_attr.name
                edge_data[i] = edge_data[i] / edge_attrs_benchmark[attr_name]
        else:
            for e_attr, e_attr_data in zip(e_attrs, edge_data):
                edge_attrs_benchmark[e_attr.name] = e_attr_data.max()
            for i, n_attr in enumerate(e_attrs):
                edge_data[i] = edge_data[i] / edge_attrs_benchmark[n_attr.name]
        return edge_data, edge_attrs_benchmark

    def get_edge_aggr_attrs_obs(self, network, edge_attr_types=['resource', 'extrema'], normalization=True, benchmark=None):
        e_attrs = network.get_edge_attrs(edge_attr_types)
        edge_aggr_attrs_data = np.array(network.get_aggregation_attrs_data(e_attrs), dtype=np.float32)

        if not normalization:
            return edge_aggr_attrs_data, {}

        if benchmark is not None:
            for i, e_attr in enumerate(e_attrs):
                attr_name = e_attr.originator if e_attr.type == 'extrema' else e_attr.name
                edge_aggr_attrs_data[i] = edge_aggr_attrs_data[i] / benchmark[attr_name]
            return edge_aggr_attrs_data, benchmark

        edge_aggr_attrs_benchmark = {}
        if 'extrema' in edge_attr_types:
            for e_attr, e_aggr_data in zip(e_attrs, edge_aggr_attrs_data):
                if e_attr.type == 'extrema':
                    edge_aggr_attrs_benchmark[e_attr.originator] = e_aggr_data.max()
            for i, e_attr in enumerate(e_attrs):
                attr_name = e_attr.originator if e_attr.type == 'extrema' else e_attr.name
                edge_aggr_attrs_data[i] = edge_aggr_attrs_data[i] / edge_aggr_attrs_benchmark[attr_name]
        else:
            for e_attr, e_attr_data in zip(e_attrs, edge_aggr_attrs_data):
                edge_aggr_attrs_benchmark[e_attr.name] = e_attr_data.max()
            for i, n_attr in enumerate(e_attrs):
                edge_aggr_attrs_data[i] = edge_aggr_attrs_data[i] / edge_aggr_attrs_benchmark[n_attr.name]
        return edge_aggr_attrs_data, edge_aggr_attrs_benchmark

    def get_edge_index_obs(self, network):
        edge_index = np.array(list(network.edges), dtype=np.int64)
        return edge_index.T

    def get_edge_pair_obs(self, network):
        return np.array(list(network.edges), dtype=np.int64)

    def get_selected_node_mask(self, network, selected_nodes):
        selected_node_mask = np.zeros(network.num_nodes, dtype=np.float32)
        selected_node_mask[selected_nodes] = 1.
        return np.array([selected_node_mask], dtype=np.float32, normalization=True)

    def get_average_distance(self, network, selected_nodes, normalization=True):
        # avg_dst
        if len(selected_nodes) == 0: 
            avg_distance = np.zeros(network.num_nodes)
        else:
            distance_dict = dict(nx.shortest_path_length(network))
            avg_distance = []
            for u in range(network.num_nodes):
                sum_dst = 0
                for v in selected_nodes:
                    sum_dst += distance_dict[u][v]
                sum_dst /= (len(selected_nodes) + 1)
                avg_distance.append(sum_dst)
            avg_distance = np.array(avg_distance)
        if normalization:
            if np.max(avg_distance) == 0:
                 avg_distance = avg_distance
            else:
            # elif np.max(avg_distance) == np.min(avg_distance):
                # avg_distance = avg_distance / np.max(avg_distance)
            # else:
                avg_distance = (avg_distance - np.min(avg_distance)) / (np.max(avg_distance) - np.min(avg_distance))
        return np.array([avg_distance], dtype=np.float32).T

    def get_average_distance_for_v_node(self, pn, vn, nodes_slot, v_node_id=None):
        distance_dict = dict(nx.shortest_path_length(self.pn))
        avg_distance = []
        for u in range(self.pn.num_nodes):
            sum_dst = 0
            for v in self.selected_pn_nodes:
                sum_dst += distance_dict[u][v]
            sum_dst /= (len(self.selected_pn_nodes) + 1)
            avg_distance.append(sum_dst)
        avg_distance = (avg_distance - np.min(avg_distance)) / (np.max(avg_distance) - np.min(avg_distance))

        # avg_dst
        if v_node_id is None or len(nodes_slot):
            avg_distance = np.zeros(pn.num_nodes)
        else:
            distance_dict = dict(nx.shortest_path_length(pn))
            avg_distance = []
            for p_u in range(pn.num_nodes):
                sum_dst = 0
                for p_v, v_n in nodes_slot.items():
                    if v_n not in list(vn.adj[v_node_id]):
                        pass
                    sum_dst += distance_dict[p_u][p_v]
                sum_dst /= (len(pn) + 1)
                avg_distance.append(sum_dst)
            if np.max(avg_distance) == np.min(avg_distance):
                avg_distance = np.zeros(pn.num_nodes)
            else:   
                avg_distance = (avg_distance - np.min(avg_distance)) / (np.max(avg_distance) - np.min(avg_distance))
        return np.array([avg_distance], dtype=np.float32)

    def get_v_node_edge_demands(self, vn, v_node_id, normalization=True, benchmark=None):
        e_attrs = vn.get_edge_attrs('resource')
        edge_demands = []
        for e_attr in e_attrs:
            edge_demand = [vn.edges[(n, v_node_id)][e_attr.name] for n in vn.adj[v_node_id]]
            edge_demands.append(edge_demand)
            # max_edge_demand.append(max(edge_demand) / self.edge_attrs_benchmark[e_attr.name])
            # mean_edge_demand.append((sum(edge_demand) / len(edge_demand)) / self.edge_attrs_benchmark[e_attr.name])
        
        edge_demands = np.array(edge_demands, dtype=np.float32)
        if not normalization:
            return edge_demands, {}

        if benchmark is not None:
            for i, e_attr in enumerate(e_attrs):
                attr_name = e_attr.name
                edge_demands[i] = edge_demands[i] / benchmark[attr_name]
            return edge_demands, benchmark

    def get_v_node_features(self, vn, v_node_id):
        if v_node_id >= vn.num_nodes:
            v_node_id = 0
            return self.get_v_node_features(vn, v_node_id)
        norm_unplaced = (vn.num_nodes - (v_node_id + 1)) / vn.num_nodes
        norm_all_nodes = vn.num_nodes / self.pn.num_nodes
        norm_curr_vid = (v_node_id + 1) / self.pn.num_nodes
        node_demand = []
        for n_attr in vn.get_node_attrs('resource'):
            node_demand.append(vn.nodes[v_node_id][n_attr.name] / self.node_attrs_benchmark[n_attr.name])
        norm_node_demand = np.array(node_demand, dtype=np.float32)

        max_edge_demand = []
        mean_edge_demand = []
        num_neighbors = len(vn.adj[v_node_id]) / vn.num_nodes
        for e_attr in vn.get_edge_attrs('resource'):
            edge_demand = [vn.edges[(n, v_node_id)][e_attr.name] for n in vn.adj[v_node_id]]
            max_edge_demand.append(max(edge_demand) / self.edge_attrs_benchmark[e_attr.name])
            mean_edge_demand.append((sum(edge_demand) / len(edge_demand)) / self.edge_attrs_benchmark[e_attr.name])

        vn_obs = np.concatenate([norm_node_demand, max_edge_demand, mean_edge_demand, [num_neighbors, norm_unplaced, norm_all_nodes, norm_curr_vid]], axis=0)
        return vn_obs