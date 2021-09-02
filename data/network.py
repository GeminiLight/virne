import ast
import copy
import numpy as np
import networkx as nx
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from typing import *
from data.utils import get_items, dict_to, path_to_edges


class Network(nx.Graph):
    r"""
    Args:
        incoming_graph_data (Graph): nx.Graph Object to intialize.

    Attributes:
        nodes (dict): Infomation of nodes.
        edges (dict): Infomation of edges.
        num_nodes (int): Number of nodes.
        num_edges (int): Number of edges.
        num_node_attrs (int): Number of node attributes.
        num_edge_attrs (int): Number of edge attributes.
        node_attrs (list): List of all node attributes.
        edge_attrs (list): List of all edge attributes.
    
    Methods:
        generate_topology: 
        generate_data: 

        is_node_attr: 
        is_edge_attr: 
        set_node_attr: 
        set_edge_attr: 
        set_network_attr: 
        set_network_attrs: 
        get_node_attr: 
        get_edge_attr: 
        get_network_attr: 

        get_attrs_data:
        get_nodes_data:
        get_edges_data: 

        update_node:
        update_edge:

        find_shortest_path: 
        find_sample_path:
    """
    def __init__(self, incoming_graph_data=None, **kwargs):
        super(Network, self).__init__(incoming_graph_data)
        # Read extra kwargs
        self.set_network_attrs(kwargs)

    def generate_topology(self, num_nodes, type='path', **kwargs):
        r"""Generate the physical network's topology according to 
            the structure type and number of nodes."""
        if type == 'path':
            G = nx.path_graph(num_nodes)
        if type == 'waxman':
            wm_alpha = kwargs.pop('wm_alpha', 0.5)
            wm_beta = kwargs.pop('wm_beta', 0.2)
            G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta)
        for key, value in G.__dict__.items():
            self.__dict__[key] = value

    def generate_data(self, node_attrs=[], edge_attrs=[], 
                        min_node_value=2, max_node_value=30, 
                        min_edge_value=2, max_edge_value=30, **kwargs):
        r"""Generate the physical network's data follow the uniform distribution from min_value to max_value.
        
        Args:
        """
        keys_set = kwargs.keys()
        # generate nodes' data
        nodes_data = np.random.randint(min_node_value, max_node_value, size=(len(node_attrs), self.num_nodes))
        for i, n_attr_name in enumerate(node_attrs):
            self.set_node_attr(n_attr_name, nodes_data[i])
        # generate edges' data
        edges_data = np.random.randint(min_edge_value, max_edge_value, size=(len(edge_attrs), self.num_edges))
        for i, e_attr_name in enumerate(edge_attrs):
            self.set_edge_attr(e_attr_name, edges_data[i])

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
    def num_node_attrs(self):
        r"""Return the number of node attributes."""
        return len(self.node_attrs)

    @property
    def num_edge_attrs(self):
        r"""Return the number of edge attributes."""
        return len(self.edge_attrs)

    ### Matrix ###
    @property
    def adjacency_matrix(self):
        r"""Return the adjacency matrix of Network."""
        return nx.to_scipy_sparse_matrix(self, format='csr')

    ### Atribute ###
    @property
    def node_attrs(self):
        r"""Return the list of all node attributes."""
        return list(self.nodes[0].keys())

    @property
    def edge_attrs(self):
        r"""Return the list of all edge attributes."""
        return list(self.edges[list(self.edges)[0]].keys())

    def is_node_attr(self, name: str):
        r"""Check whether an attribute is a node attribute."""
        return name in self.node_attrs

    def is_edge_attr(self, name: str):
        r"""
        Check whether an attribute is a edge attribute.
        """
        return name in self.edge_attrs

    def set_network_attr(self, attr: Union[dict, str], value):
        r"""Set graph attribute `attr` to `value`.

        Args:
            attr (str): the attribute name or dictionary. if the type of attr is str, 
                        additional value should be provided.
            value (optional): the attribute value that should be given 
                        when the type of attr is str.
        """
        if isinstance(attr, dict):
            attr, value = list(attr.items())
        self.graph[attr] = value
        self[attr] = value

    def set_network_attrs(self, attrs: Union[dict, list], values: list = None):
        r"""Set graph attributes.

        Args:
            attr (str): the attribute name or dictionary. if the type of attr is str, 
                        additional value should be provided.
            value (optional): the attribute value that should be given 
                        when the type of attr is str.
        """
        items = get_items(attrs, values)
        for key, value in attrs.items():
            self.set_network_attr(key, value)
                
    def set_node_attr(self, name: str, value: Union[dict, list]):
        r"""Set node attribute `name` to `value`."""
        if isinstance(value, dict):
            nx.set_node_attributes(self, value, name)
        elif isinstance(value, (list, np.ndarray)):
            value_dict = {i: value[i] for i in range(self.num_nodes)}
            nx.set_node_attributes(self, value_dict, name)

    def set_edge_attr(self, name: str, value: Union[dict, list, np.ndarray]):
        r"""Set edge attribute `name` to `value`."""
        if isinstance(value, dict):
            nx.set_edge_attributes(self, value, name)
        elif isinstance(value, (list, np.ndarray)):
            value_dict = {e: value[i] for i, e in enumerate(self.edges)}
            nx.set_edge_attributes(self, value_dict, name)

    def get_node_attr(self, name: str, rtype: str = 'list'):
        r"""Get the attribute value of all node returned in the appropriate format.
        
        Args:
            name (str): Name of attribute.
            rtype (str): Format of returned data, default='list'.
        """
        attr_dict = nx.get_node_attributes(self, name)
        return dict_to(attr_dict, rtype)

    def get_edge_attr(self, name: str, rtype: str ='list'):
        r"""Get the attribute value of all edge returned in the appropriate format.
        
        Args:
            name (str): Name of attribute.
            rtype (str): Format of returned data, default='list'.
        """
        attr_dict = nx.get_edge_attributes(self, name)
        return dict_to(attr_dict, rtype)

    def get_graph_attr(self, key: str):
        r"""Returns the graph attributes.
        
        Args:
            key(string): the name of the attributes to return.
        Returns:
            any: graph attributes with the specified name.
        """
        return self.graph[key]

    def get_attrs_data(self, attrs: list):
        r"""Return the given attributes data.
        
        Args:
            attrs: (list): List of attribute names.
        """
        attrs_data = []
        for attr in attrs:
            if self.is_node_attr(attr):
                attr_data = list(nx.get_node_attributes(self, attr).values())
            elif self.is_edge_attr(attr):
                attr_data = list(nx.get_edge_attributes(self, attr).values())
            else:
                raise Exception("No such attribute")
            attrs_data.append(attr_data)
        return attrs_data

    ### Data ###
    def get_nodes_data(self, node_attrs: list):
        r"""Return the data of nodes according to the given attributes.
        
        Args:
            node_attrs: (list): List of node attribute names.
        """
        if not isinstance(node_attrs, list):
            raise TypeError("the type of arguments `node_attrs` should be 'list'")
        for attr in node_attrs:
            if not self.is_node_attr(attr):
                raise Exception("No such node attribute")
        return self.get_attrs_data(node_attrs)

    def get_edges_data(self, edge_attrs: list):
        r"""Return the data of nodes according to the given attributes.
        
        Args:
            edge_attrs: (list): List of node attribute names.
        """
        if not isinstance(edge_attrs, list):
            raise TypeError("the type of arg 'edge_attrs' should be 'list'")
        for attr in edge_attrs:
            if not self.is_edge_attr(attr):
                raise Exception("No such edge attribute")
        return self.get_attrs_data(edge_attrs)

    ### Update ###
    def update_node(self, node, update_attrs, increments=None):
        r"""Update (increase) the value of node atributes."""
        items = get_items(update_attrs, increments)
        for attr, increment in items:
            self.nodes[node][attr] += increment

    def update_edge(self, edge, update_attrs, increments=None):
        r"""Update (increase) the value of edge atributes."""
        items = get_items(update_attrs, increments)
        for attr, increment in update_attrs.items():
            self.edges[edge][attr] += increment

    def update_path(self, path, update_attrs, increments=None):
        r"""Update (increase) the value of edges atributes of path with the same increments."""
        if len(path) == 1:
            return True
        edges_list = path_to_edges(path)
        for edge in edges_list:
            self.update_edge(edge, update_attrs)

    ### Algo ###
    def find_simple_paths(self, i, j):
        r"""Find the shortest simple paths of node `i` and `j`."""
        try:
            simple_paths = nx.shortest_simple_paths(self, i, j)
            return simple_paths
        except:
            return []

    def find_shortest_path(self, i, j):
        r"""Find the shortest path (dijkstra) of node a and b."""
        # try:
        shortest_path = nx.dijkstra_path(self, i, j)
        return shortest_path
        # except:
            # return False

    ### IO ###
    @classmethod
    def from_graph(cls, G):
        net = cls()
        for key, value in G.__dict__.items():
            net.__dict__[key] = value
        return net

    @classmethod
    def from_csv(cls, node_path, edge_path):
        nodes_data = pd.read_csv(node_path, index_col=[0])
        edges_data = pd.read_csv(edge_path, index_col=[0, 1])
        net = cls()
        net.add_nodes_from(nodes_for_adding=list(nodes_data.index))
        net.add_edges_from(edges_data.index)
        for n_attr_name in nodes_data.columns:
            n_attr_data = nodes_data[n_attr_name].values
            if isinstance(n_attr_data[0], str):
                n_attr_data = [ast.literal_eval(v) for v in n_attr_data]
            net.set_node_attr(n_attr_name, n_attr_data)
        for e_attr_name in edges_data.columns:
            e_attr_data = edges_data[e_attr_name].values
            if isinstance(e_attr_data[0], str):
                e_attr_data = [ast.literal_eval(v) for v in e_attr_data]
            net.set_edge_attr(e_attr_name, e_attr_data)
        return net

    def to_csv(self, node_path, edge_path):
        r""""Save the node data and edge data of physical network to `node_path` and `edge_path`, respectively."""
        nodes_data = {}
        for attr in self.node_attrs:
            attr_data = self.get_node_attr(attr)
            nodes_data[attr] = attr_data
        nodes = pd.DataFrame(nodes_data, index=self.nodes)
        nodes.to_csv(node_path)
        edges_data = {}
        for attr in self.edge_attrs:
            attr_data = self.get_edge_attr(attr)
            edges_data[attr] = attr_data
        edges = pd.DataFrame(edges_data, index=self.edges)
        edges.to_csv(edge_path)

    def draw(self, show=True, save_path=None):
        nx.draw(self)
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    ### Internal ###
    # def __repr__(self):
    #     info = [f"{key}={self._size_repr(item)}" for key, item in self]
    #     return f"{self.__class__.__name__}({', '.join(info)})"

    def __getitem__(self, key):
        r"""Gets the data of the attribute key."""
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, str):
            return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        r"""Sets the attribute key to value."""
        setattr(self, key, value)

    def clone(self):
        r"""Deepcopy the network object.

        Returns:
            Network (class): A cloned Network object with deepcopying all features.
        """
        return self.__class__.from_dict({
            k: copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })


if __name__ == '__main__':
    num_nodes = 10
    net = Network.from_graph(nx.path_graph(num_nodes))
    net.draw()
    
    cpu_data = np.random.randint(net.num_nodes, size=(10))
    cpu = {i: cpu_data[i] for i in range(num_nodes)}
    print(cpu)

    # def generate_topology(self, num_nodes, type='path', **kwargs):
    #     if type == 'path':
    #         G = nx.path_graph(num_nodes)
    #     elif type == 'waxman':
    #         G = nx.waxman_graph(num_nodes, kwargs.pop('alpha', 0.5), kwargs.pop('beta', 0.2))
    #         self.set_node_attr({'name': 'pos'})
    #     elif type == 'random':
    #         G = nx.fast_gnp_random_graph(num_nodes, kwargs.pop('prob', 0.5))
    #     self.from_graph(G)

    # def generate_data_for_attr(self, attr, num):
    #     r"""Generate data according to the settings of attributes."""
    #     if attr['type'] in ['c', 'con', 'congruent']:
    #         attr_data = np.array(attr['value'] * num)
    #     elif attr['type'] in ['o', 'or', 'order']:
    #         attr_data = np.arange(num) + attr['value'][0]
    #     elif attr['type'] in ['r', 'ran', 'range']:
    #         attr_data = np.arange(attr['value'][0], attr['value'][1], attr['value'][2])
    #     elif attr['type'] in ['l', 'lin', 'linspace']:
    #         attr_data = np.linspace(attr['value'][0], attr['value'][1], attr['value'][2])
    #     elif attr['type'] in ['e', 'ex', 'extremum']:
    #         attr_data = np.random.randint(attr['value'][0], attr['value'][1]+1, num)
    #     elif attr['type'] in ['i', 'in', 'inclusion']:
    #         attr_data = np.random.choice(attr['value'], num, p=attr['other'])
    #     else:
    #         raise AttributeError("The type of attribution isn't assigned before genaterating data")

    # def generate_data(self):
    #     for attr in self.node_attrs:
    #         if not attr.has_key('value'):
    #             continue
    #         attr_data = self.generate_data_for_attr(attr, self.num_nodes)
    #     for attr in self.edge_attrs:
    #         if not attr.has_key('value'):
    #             continue
    #         attr_data = self.generate_data_for_attr(attr, self.num_edges)