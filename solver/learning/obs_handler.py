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

    def get_link_attrs_obs(self, network, link_attr_types=['resource', 'extrema'], normalization=True, benchmark=None):
        e_attrs = network.get_link_attrs(link_attr_types)
        link_data = np.array(network.get_link_attrs_data(e_attrs), dtype=np.float32)
        link_data = np.concatenate([link_data, link_data], axis=1)
        if not normalization:
            return link_data, {}

        if benchmark is not None:
            for i, l_attr in enumerate(e_attrs):
                attr_name = l_attr.originator if l_attr.type == 'extrema' else l_attr.name
                link_data[i] = link_data[i] / benchmark[attr_name]
            return link_data, benchmark

        link_attrs_benchmark = {}
        if 'extrema' in link_attr_types:
            for l_attr, e_attr_data in zip(e_attrs, link_data):
                if l_attr.type == 'extrema':
                    link_attrs_benchmark[l_attr.originator] = e_attr_data.max()
            for i, l_attr in enumerate(e_attrs):
                attr_name = l_attr.originator if l_attr.type == 'extrema' else l_attr.name
                link_data[i] = link_data[i] / link_attrs_benchmark[attr_name]
        else:
            for l_attr, e_attr_data in zip(e_attrs, link_data):
                link_attrs_benchmark[l_attr.name] = e_attr_data.max()
            for i, n_attr in enumerate(e_attrs):
                link_data[i] = link_data[i] / link_attrs_benchmark[n_attr.name]
        return link_data, link_attrs_benchmark

    def get_link_aggr_attrs_obs(self, network, link_attr_types=['resource', 'extrema'], normalization=True, benchmark=None):
        e_attrs = network.get_link_attrs(link_attr_types)
        link_aggr_attrs_data = np.array(network.get_aggregation_attrs_data(e_attrs), dtype=np.float32)

        if not normalization:
            return link_aggr_attrs_data, {}

        if benchmark is not None:
            for i, l_attr in enumerate(e_attrs):
                attr_name = l_attr.originator if l_attr.type == 'extrema' else l_attr.name
                link_aggr_attrs_data[i] = link_aggr_attrs_data[i] / benchmark[attr_name]
            return link_aggr_attrs_data, benchmark

        link_aggr_attrs_benchmark = {}
        if 'extrema' in link_attr_types:
            for l_attr, e_aggr_data in zip(e_attrs, link_aggr_attrs_data):
                if l_attr.type == 'extrema':
                    link_aggr_attrs_benchmark[l_attr.originator] = e_aggr_data.max()
            for i, l_attr in enumerate(e_attrs):
                attr_name = l_attr.originator if l_attr.type == 'extrema' else l_attr.name
                link_aggr_attrs_data[i] = link_aggr_attrs_data[i] / link_aggr_attrs_benchmark[attr_name]
        else:
            for l_attr, e_attr_data in zip(e_attrs, link_aggr_attrs_data):
                link_aggr_attrs_benchmark[l_attr.name] = e_attr_data.max()
            for i, n_attr in enumerate(e_attrs):
                link_aggr_attrs_data[i] = link_aggr_attrs_data[i] / link_aggr_attrs_benchmark[n_attr.name]
        return link_aggr_attrs_data, link_aggr_attrs_benchmark

    def get_link_index_obs(self, network):
        link_index = np.array(list(network.links), dtype=np.int64)
        link_index = np.concatenate([link_index, link_index[:, [1,0]]], axis=0)
        return link_index.T

    def get_link_pair_obs(self, network):
        return np.array(list(network.links), dtype=np.int64)

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

    def get_average_distance_for_v_node(self, p_net, v_net, nodes_slot, v_node_id=None):
        distance_dict = dict(nx.shortest_path_length(self.p_net))
        avg_distance = []
        for u in range(self.p_net.num_nodes):
            sum_dst = 0
            for v in self.selected_p_net_nodes:
                sum_dst += distance_dict[u][v]
            sum_dst /= (len(self.selected_p_net_nodes) + 1)
            avg_distance.append(sum_dst)
        avg_distance = (avg_distance - np.min(avg_distance)) / (np.max(avg_distance) - np.min(avg_distance))

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
            if np.max(avg_distance) == np.min(avg_distance):
                avg_distance = np.zeros(p_net.num_nodes)
            else:   
                avg_distance = (avg_distance - np.min(avg_distance)) / (np.max(avg_distance) - np.min(avg_distance))
        return np.array([avg_distance], dtype=np.float32)

    def get_v_node_link_demands(self, v_net, v_node_id, normalization=True, benchmark=None):
        e_attrs = v_net.get_link_attrs('resource')
        link_demands = []
        for l_attr in e_attrs:
            link_demand = [v_net.links[(n, v_node_id)][l_attr.name] for n in v_net.adj[v_node_id]]
            link_demands.append(link_demand)
            # max_link_demand.append(max(link_demand) / self.link_attrs_benchmark[l_attr.name])
            # mean_link_demand.append((sum(link_demand) / len(link_demand)) / self.link_attrs_benchmark[l_attr.name])
        
        link_demands = np.array(link_demands, dtype=np.float32)
        if not normalization:
            return link_demands, {}

        if benchmark is not None:
            for i, l_attr in enumerate(e_attrs):
                attr_name = l_attr.name
                link_demands[i] = link_demands[i] / benchmark[attr_name]
            return link_demands, benchmark

    def get_v_node_features(self, v_net, v_node_id):
        if v_node_id >= v_net.num_nodes:
            v_node_id = 0
            return self.get_v_node_features(v_net, v_node_id)
        norm_unplaced = (v_net.num_nodes - (v_node_id + 1)) / v_net.num_nodes
        norm_all_nodes = v_net.num_nodes / self.p_net.num_nodes
        norm_curr_vid = (v_node_id + 1) / self.p_net.num_nodes
        node_demand = []
        for n_attr in v_net.get_node_attrs('resource'):
            node_demand.append(v_net.nodes[v_node_id][n_attr.name] / self.node_attrs_benchmark[n_attr.name])
        norm_node_demand = np.array(node_demand, dtype=np.float32)

        max_link_demand = []
        mean_link_demand = []
        num_neighbors = len(v_net.adj[v_node_id]) / v_net.num_nodes
        for l_attr in v_net.get_link_attrs('resource'):
            link_demand = [v_net.links[(n, v_node_id)][l_attr.name] for n in v_net.adj[v_node_id]]
            max_link_demand.append(max(link_demand) / self.link_attrs_benchmark[l_attr.name])
            mean_link_demand.append((sum(link_demand) / len(link_demand)) / self.link_attrs_benchmark[l_attr.name])

        v_net_obs = np.concatenate([norm_node_demand, max_link_demand, mean_link_demand, [num_neighbors, norm_unplaced, norm_all_nodes, norm_curr_vid]], axis=0)
        return v_net_obs