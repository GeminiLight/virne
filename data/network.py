import copy
import networkx as nx

from .utils import write_setting

from .attribute import Attribute


class Network(nx.Graph):

    def __init__(self, incoming_graph_data=None, node_attrs_setting=[], edge_attrs_setting=[], **kwargs):
        super(Network, self).__init__(incoming_graph_data)
        self.init_graph_attrs()
        # set graph attributes
        self.graph['node_attrs_setting'] += node_attrs_setting
        self.graph['edge_attrs_setting'] += edge_attrs_setting
        self.create_attrs_from_setting()
        # Read extra kwargs
        self.set_graph_attrs_data(kwargs)

    def create_attrs_from_setting(self):
        self.node_attrs = {n_attr_dict['name']: Attribute.from_dict(n_attr_dict) for n_attr_dict in self.graph['node_attrs_setting']}
        self.edge_attrs = {e_attr_dict['name']: Attribute.from_dict(e_attr_dict) for e_attr_dict in self.graph['edge_attrs_setting']}

    def check_attrs_existence(self):
        # check node attrs
        for n_attr_name in self.node_attrs.keys():
            assert n_attr_name in self.nodes[list(self.nodes)[0]].keys(), f'{n_attr_name}'
        # check edge attrs
        for e_attr_name in self.edge_attrs.keys():
            assert e_attr_name in self.edges[list(self.edges)[0]].keys(), f'{e_attr_name}'

    ### Generate ###
    def generate_topology(self, num_nodes, type='path', **kwargs):
        r"""Generate the network topology."""
        assert type in ['path', 'waxman', 'random']
        self.set_graph_attrs_data({'num_nodes': num_nodes, 'type': type})
        if type == 'path':
            G = nx.path_graph(num_nodes)
        elif type == 'waxman':
            wm_alpha = kwargs.get('wm_alpha', 0.5)
            wm_beta = kwargs.get('wm_beta', 0.2)
            not_connected = True
            while not_connected:
                G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta)
                not_connected = not nx.is_connected(G)
            self.set_graph_attrs_data({'wm_alpha': wm_alpha, 'wm_beta': wm_beta})
        elif type == 'random':
            random_prob = kwargs.get('random_prob', 0.5)
            self.set_graph_attrs_data({'random_prob': random_prob})
            G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
            not_connected = True
            while not_connected:
                G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
                not_connected = not nx.is_connected(G)
        else:
            raise NotImplementedError
        self.__dict__['_node'] = G.__dict__['_node']
        self.__dict__['_adj'] = G.__dict__['_adj']

    def generate_attrs_data(self, node=True, edge=True):
        r"""Generate the data of network attributes based on attributes."""
        if node:
            for n_attr in self.node_attrs.values():
                if n_attr.generative or n_attr.type == 'extrema':
                    attribute_data = n_attr.generate_data(self)
                    n_attr.set_data(self, attribute_data)
        if edge:
            for e_attr in self.edge_attrs.values():
                if e_attr.generative or n_attr.type == 'extrema':
                    attribute_data = e_attr.generate_data(self)
                    e_attr.set_data(self, attribute_data)

    ### Number ###
    @property
    def num_nodes(self):
        r"""Return the number of nodes."""
        return self.number_of_nodes()
    
    @property
    def num_edges(self):
        r"""Return the number of edges."""
        return self.number_of_edges()

    @property
    def adjacency_matrix(self):
        r"""Return the adjacency matrix of Network."""
        return nx.to_scipy_sparse_matrix(self, format='csr')

    ### Get Attribute ###
    def get_graph_attrs(self, attrs=None):
        if attrs is None: return self.graph
        return {attr: self.graph[attr] for attr in attrs}

    def get_node_attrs(self, types=None):
        if types is None: return list(self.node_attrs.values())
        elif isinstance(type, list):
            types = [type]
        selected_node_attrs = []
        for n_attr in self.node_attrs.values():
            selected_node_attrs.append(n_attr) if n_attr.type in types else None
        return selected_node_attrs

    def get_edge_attrs(self, types=None):
        if types is None: return list(self.edge_attrs.values())
        if isinstance(type, list):
            types = [type]
        selected_edge_attrs = []
        for e_attr in self.edge_attrs.values():
            selected_edge_attrs.append(e_attr) if e_attr.type in types else None
        return selected_edge_attrs

    ### Set Data ### 
    def init_graph_attrs(self):
        for key in ['node_attrs_setting', 'edge_attrs_setting']:
            if key not in self.graph:
                self.graph[key] = []
        for key, value in self.graph.items():
            if key not in ['num_nodes']:
                self[key] = value

    def set_graph_attribute(self, name, value):
        r"""Set graph attribute `attr` to `value`."""
        if name in ['num_nodes']:
            self.graph[name] = value
            return
        self.graph[name] = value
        self[name] = value

    def set_graph_attrs_data(self, attributes_data):
        r"""Set graph attributes."""
        for key, value in attributes_data.items():
            self.set_graph_attribute(key, value)

    def set_node_attrs_data(self, node_attributes_data):
        for n_attr, data in node_attributes_data.items():
            n_attr.set_data(self, data)

    def set_edge_attrs_data(self, edge_attributes_data):
        for e_attr, data in edge_attributes_data.items():
            e_attr.set_data(self, data)

    ### Get Data ###
    def get_node_attrs_data(self, node_attrs):
        node_attrs_data = [n_attr.get_data(self) for n_attr in node_attrs]
        return node_attrs_data

    def get_edge_attrs_data(self, edge_attrs):
        edge_attrs_data = [e_attr.get_data(self) for e_attr in edge_attrs]
        return edge_attrs_data

    def get_adjacency_attrs_data(self, edge_attrs, normalized=False):
        adjacency_data = [e_attr.get_adjacency_data(self, normalized) for e_attr in edge_attrs]
        return adjacency_data

    def get_aggregation_attrs_data(self, edge_attrs, aggr='sum', normalized=False):
        aggregation_data = [e_attr.get_aggregation_data(self, aggr, normalized) for e_attr in edge_attrs]
        return aggregation_data

    ### other ###
    def subgraph(self, nodes):
        subnet = super().subgraph(nodes)
        subnet.node_attrs = self.node_attrs
        subnet.edge_attrs = self.edge_attrs
        return subnet

    ### Constraint ###
    def check_node_constraints(self, p_node_id, vn, v_node_id):
        for n_attr in vn.node_attrs.values():
            if not n_attr.check(vn.nodes[v_node_id], self.nodes[p_node_id]):
                return False
        return True

    ### Update ###
    def update_node_resources(self, node_id, vn_node, method='+'):
        r"""Update (increase) the value of node atributes."""
        for n_attr in self.node_attrs.keys():
            if n_attr.type != 'resource':
                continue
            n_attr.update(self.nodes[node_id], vn_node, method)

    def update_edge_resources(self, edge_pair, vn_edge, method='+'):
        r"""Update (increase) the value of edge atributes."""
        for e_attr in self.edge_attrs:
            if e_attr.type != 'resource':
                continue
            e_attr.update(self.edges[edge_pair], vn_edge, method)

    def update_path_resources(self, path, vn_edge, method='+'):
        r"""Update (increase) the value of edges atributes of path with the same increments."""
        assert len(path) >= 1
        for e_attr in self.edge_attrs:
            e_attr.update_path(self, path, vn_edge, method)

    ### Internal ###
    def __getitem__(self, key):
        r"""Gets the data of the attribute key."""
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, str):
            return getattr(self, key, None)
        else:
            return TypeError

    # def __repr__(self):
    #     info = [f"{key}={self._size_repr(item)}" for key, item in self]
    #     return f"{self.__class__.__name__}({', '.join(info)})"

    def __setitem__(self, key: str, value):
        r"""Sets the attribute key to value."""
        setattr(self, key, value)

    def clone(self):
        return self.__class__.from_dict({
            k: copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })

    def to_gml(self, fpath):
        nx.write_gml(self, fpath)

    @classmethod
    def from_gml(cls, fpath):
        gml_net = nx.read_gml(fpath, destringizer=int)
        net = cls(incoming_graph_data=gml_net)
        net.check_attrs_existence()
        return net

    def save_attrs_dict(self, fpath):
        attrs_dict = {
            'graph_attrs_dict': self.get_graph_attrs(),
            'node_attrs': [n_attr.to_dict() for n_attr in self.node_attrs.values()],
            'edge_attrs': [e_attr.to_dict() for e_attr in self.edge_attrs.values()]
        }
        write_setting(attrs_dict, fpath)


if __name__ == '__main__':
    pass
