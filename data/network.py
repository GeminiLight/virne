import copy
import networkx as nx

from utils import write_setting
from .attribute import Attribute
from functools import cached_property
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView


class Network(nx.Graph):

    def __init__(self, incoming_graph_data=None, node_attrs_setting=[], link_attrs_setting=[], **kwargs):
        super(Network, self).__init__(incoming_graph_data)
        self.init_graph_attrs()
        # set graph attributes
        self.graph['node_attrs_setting'] += node_attrs_setting
        self.graph['link_attrs_setting'] += link_attrs_setting
        self.create_attrs_from_setting()
        # Read extra kwargs
        self.set_graph_attrs_data(kwargs)

    def create_attrs_from_setting(self):
        self.node_attrs = {n_attr_dict['name']: Attribute.from_dict(n_attr_dict) for n_attr_dict in self.graph['node_attrs_setting']}
        self.link_attrs = {e_attr_dict['name']: Attribute.from_dict(e_attr_dict) for e_attr_dict in self.graph['link_attrs_setting']}

    def check_attrs_existence(self):
        # check node attrs
        for n_attr_name in self.node_attrs.keys():
            assert n_attr_name in self.nodes[list(self.nodes)[0]].keys(), f'{n_attr_name}'
        # check link attrs
        for l_attr_name in self.link_attrs.keys():
            assert l_attr_name in self.links[list(self.links)[0]].keys(), f'{l_attr_name}'

    ### Generate ###
    def generate_topology(self, num_nodes, type='path', **kwargs):
        r"""Generate the network topology."""
        assert num_nodes >= 1
        assert type in ['path', 'star', 'waxman', 'random'], ValueError('Unsupported graph type!')
        self.set_graph_attrs_data({'num_nodes': num_nodes, 'type': type})
        if type == 'path':
            G = nx.path_graph(num_nodes)
        elif type == 'star':
            G = nx.star_graph(num_nodes)
        elif type == 'grid_2d':
            wm_alpha = kwargs.get('m')
            wm_beta = kwargs.get('n')
            G = nx.grid_2d_graph(num_nodes, periodic=False)
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

    def generate_attrs_data(self, node=True, link=True):
        r"""Generate the data of network attributes based on attributes."""
        if node:
            for n_attr in self.node_attrs.values():
                if n_attr.generative or n_attr.type == 'extrema':
                    attribute_data = n_attr.generate_data(self)
                    n_attr.set_data(self, attribute_data)
        if link:
            for l_attr in self.link_attrs.values():
                if l_attr.generative or l_attr.type == 'extrema':
                    attribute_data = l_attr.generate_data(self)
                    l_attr.set_data(self, attribute_data)

    ### Number ###
    @property
    def num_nodes(self):
        r"""Return the number of nodes."""
        return self.number_of_nodes()
    
    @property
    def num_links(self):
        r"""Return the number of links."""
        return self.number_of_edges()

    @cached_property
    def links(self):
        r"""Return the number of links."""
        return EdgeView(self)


    @property
    def adjacency_matrix(self):
        r"""Return the adjacency matrix of Network."""
        return nx.to_scipy_sparse_matrix(self, format='csr')

    ### Get Attribute ###
    def get_graph_attrs(self, names):
        if names is None: return self.graph
        return {attr: self.graph[attr] for attr in names}

    def get_node_attrs(self, types=None, names=None):
        if types is None and names is None: 
            return list(self.node_attrs.values())
        elif types is not None:  
            selected_node_attrs = []
            for n_attr in self.node_attrs.values():
                selected_node_attrs.append(n_attr) if n_attr.type in types else None
        elif names is not None:  
            selected_node_attrs = []
            for n_attr in self.node_attrs.values():
                selected_node_attrs.append(n_attr) if n_attr.name in names else None
        return selected_node_attrs

    def get_link_attrs(self, types=None, names=None):
        if types is None and names is None: 
            return list(self.link_attrs.values())
        elif types is not None:  
            selected_link_attrs = []
            for l_attr in self.link_attrs.values():
                selected_link_attrs.append(l_attr) if l_attr.type in types else None
        elif names is not None:  
            selected_link_attrs = []
            for l_attr in self.link_attrs.values():
                selected_link_attrs.append(l_attr) if l_attr.name in names else None
        return selected_link_attrs

    ### Set Data ### 
    def init_graph_attrs(self):
        for key in ['node_attrs_setting', 'link_attrs_setting']:
            if key not in self.graph:
                self.graph[key] = []
        for key, value in self.graph.items():
            if key not in ['num_nodes']:
                self[key] = value

    def set_graph_attribute(self, name, value):
        r"""Set a graph attribute `attr` to `value`."""
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

    def set_link_attrs_data(self, link_attributes_data):
        for l_attr, data in link_attributes_data.items():
            l_attr.set_data(self, data)

    ### Get Data ###
    def get_node_attrs_data(self, node_attrs):
        if isinstance(node_attrs[0], str):
            node_attrs_data = [list(nx.get_node_attributes(self, n_attr_name).values()) for n_attr_name in node_attrs]
        else:
            node_attrs_data = [n_attr.get_data(self) for n_attr in node_attrs]
        return node_attrs_data

    def get_link_attrs_data(self, link_attrs):
        if isinstance(link_attrs[0], str):
            link_attrs_data = [list(nx.get_edge_attributes(self, l_attr_name).values()) for l_attr_name in link_attrs]
        else:
            link_attrs_data = [l_attr.get_data(self) for l_attr in link_attrs]
        return link_attrs_data

    def get_adjacency_attrs_data(self, link_attrs, normalized=False):
        adjacency_data = [l_attr.get_adjacency_data(self, normalized) for l_attr in link_attrs]
        return adjacency_data

    def get_aggregation_attrs_data(self, link_attrs, aggr='sum', normalized=False):
        aggregation_data = [l_attr.get_aggregation_data(self, aggr, normalized) for l_attr in link_attrs]
        return aggregation_data

    ### other ###
    def subgraph(self, nodes):
        subnet = super().subgraph(nodes)
        subnet.node_attrs = self.node_attrs
        subnet.link_attrs = self.link_attrs
        return subnet

    ### Update ###
    def update_node_resources(self, node_id, v_net_node, method='+'):
        r"""Update (increase) the value of node atributes."""
        for n_attr in self.node_attrs.keys():
            if n_attr.type != 'resource':
                continue
            n_attr.update(self.nodes[node_id], v_net_node, method)

    def update_link_resources(self, link_pair, v_net_link, method='+'):
        r"""Update (increase) the value of link atributes."""
        for l_attr in self.link_attrs:
            if l_attr.type != 'resource':
                continue
            l_attr.update(self.links[link_pair], v_net_link, method)

    def update_path_resources(self, path, v_net_link, method='+'):
        r"""Update (increase) the value of links atributes of path with the same increments."""
        assert len(path) >= 1
        for l_attr in self.link_attrs:
            l_attr.update_path(self, path, v_net_link, method)

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
            'link_attrs': [l_attr.to_dict() for l_attr in self.link_attrs.values()]
        }
        write_setting(attrs_dict, fpath)


if __name__ == '__main__':
    pass