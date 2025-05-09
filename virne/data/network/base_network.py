# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import numpy as np
import networkx as nx
import logging # Added for product-level logging

from typing import Any, Dict, List, Optional, Iterable, Union, TypeVar # Added TypeVar
from functools import cached_property, lru_cache
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.classes.filters import no_filter

from virne.utils import write_setting
from ..attribute import create_attr_from_dict

logger = logging.getLogger(__name__) # Added logger

# For type hinting the class itself in class methods like from_gml
_BaseNetworkT = TypeVar('_BaseNetworkT', bound='BaseNetwork')


class BaseNetwork(nx.Graph):
    """
    Network class inherited from networkx.Graph.

    Attributes:
        node_attrs (Dict[str, Any]): Node attribute objects, keyed by attribute name.
        link_attrs (Dict[str, Any]): Link attribute objects, keyed by attribute name.
        # graph_attrs are stored in self.graph (inherited from nx.Graph)
    
    Methods:
        create_attrs_from_setting: Create node and link attribute dictionaries from their respective settings.
        init_graph_attrs: Initialize graph attributes.
        set_graph_attrs_data: Set graph attributes data.
        get_graph_attrs_data: Get graph attributes data.
        get_node_attrs: Get node attributes.
        get_link_attrs: Get link attributes.
        get_node_attrs_data: Get node attributes data.
        get_link_attrs_data: Get link attributes data.
        # Removed get_node_attr_data / get_link_attr_data from methods list as they are not defined
        set_node_attrs_data: Set node attributes data.
        set_link_attrs_data: Set link attributes data.
        # Removed set_node_attr_data / set_link_attr_data from methods list as they are not defined
        generate_topology: Generate the network topology.
        check_attrs_existence: Check if all defined attributes exist in the graph.
        write_setting: Write network setting to file.
    """
    def __init__(self, 
                 incoming_graph_data: Optional[nx.Graph] = None, 
                 node_attrs_setting: Optional[List[Dict[str, Any]]] = None, 
                 link_attrs_setting: Optional[List[Dict[str, Any]]] = None, 
                 **kwargs):
        """
        Initializes a new Network instance.

        Args:
            incoming_graph_data (optional, Graph): Graph instance to convert to Network or data to initialize graph. Default: None.
            node_attrs_setting (list): List of dictionaries containing node attribute settings. Defaults to an empty list.
            link_attrs_setting (list): List of dictionaries containing link attribute settings. Defaults to an empty list.
            **kwargs: Additional keyword arguments to set graph attributes.
        """
        super().__init__(incoming_graph_data)
        
        # Ensure node_attrs_setting and link_attrs_setting are initialized in self.graph
        # These will be populated by init_graph_attrs if not present,
        # or appended to if they are already in incoming_graph_data.graph
        if node_attrs_setting is None:
            node_attrs_setting = []
        if link_attrs_setting is None:
            link_attrs_setting = []

        self.init_graph_attrs() # Initializes self.graph['node_attrs_setting'] and ['link_attrs_setting'] if not present

        # Append provided settings to potentially existing ones (e.g., from incoming_graph_data)
        self.graph['node_attrs_setting'].extend(node_attrs_setting)
        self.graph['link_attrs_setting'].extend(link_attrs_setting)
        
        self.node_attrs: Dict[str, Any] = {}
        self.link_attrs: Dict[str, Any] = {}
        self.create_attrs_from_setting()
        
        # Read extra kwargs and set them as graph attributes
        self.set_graph_attrs_data(kwargs)

    def create_attrs_from_setting(self):
        """Create node and link attribute dictionaries from their respective settings."""
        self.node_attrs = {
            n_attr_dict['name']: create_attr_from_dict(n_attr_dict) 
            for n_attr_dict in self.graph.get('node_attrs_setting', [])
        }
        self.link_attrs = {
            e_attr_dict['name']: create_attr_from_dict(e_attr_dict) 
            for e_attr_dict in self.graph.get('link_attrs_setting', [])
        }

    def check_attrs_existence(self):
        """
        Check if all defined attributes (from settings) exist in the graph's actual node/link data.
        This is a basic check on the first node/link, assuming attribute homogeneity.
        """
        if not self.nodes:
            logger.warning("Graph has no nodes; skipping node attribute existence check.")
        elif self.node_attrs:
            first_node_id = list(self.nodes)[0]
            first_node_data = self.nodes[first_node_id]
            for n_attr_name in self.node_attrs.keys():
                if n_attr_name not in first_node_data:
                    # Consider raising an error or a more comprehensive check if strictness is required
                    logger.warning(f"Node attribute '{n_attr_name}' not found in first node {first_node_id}. Data: {first_node_data}")
                    # assert n_attr_name in first_node_data, \
                    #        f"Defined node attribute '{n_attr_name}' not found in node {first_node_id}."

        if not self.edges: # Changed from self.links to self.edges for consistency with nx.Graph
            logger.warning("Graph has no links; skipping link attribute existence check.")
        elif self.link_attrs:
            first_edge = list(self.edges)[0]
            first_edge_data = self.edges[first_edge]
            for l_attr_name in self.link_attrs.keys():
                if l_attr_name not in first_edge_data:
                    logger.warning(f"Link attribute '{l_attr_name}' not found in first link {first_edge}. Data: {first_edge_data}")
                    # assert l_attr_name in first_edge_data, \
                    #        f"Defined link attribute '{l_attr_name}' not found in edge {first_edge}."

    ### Generate ###
    def generate_topology(self, num_nodes: int, type: str = 'path', **kwargs):
        """
        Generate the network topology. This will clear any existing graph data.

        Args:
            num_nodes: The number of nodes in the generated graph.
            type: The type of graph to generate. Supported: 'path', 'star', 'waxman', 'random', 'grid_2d'.
            **kwargs: Additional keyword arguments required for certain graph types.
        """
        if not isinstance(num_nodes, int) or num_nodes < 1:
            raise ValueError("num_nodes must be a positive integer.")
        
        supported_types = ['path', 'star', 'waxman', 'random', 'grid_2d']
        if type not in supported_types:
            raise ValueError(f"Unsupported graph type: '{type}'. Supported types are: {supported_types}")

        self.set_graph_attrs_data({'num_nodes': num_nodes, 'type': type})
        
        G: nx.Graph # Type hint for G
        if type == 'path':
            G = nx.path_graph(num_nodes)
        elif type == 'star':
            # nx.star_graph(n) creates a graph with n+1 nodes.
            # Assuming num_nodes is the total number of nodes.
            G = nx.star_graph(num_nodes -1 if num_nodes > 0 else 0)
        elif type == 'grid_2d':
            # nx.grid_2d_graph(m, n) creates m*n nodes.
            # This part needs clarification on how num_nodes relates to m, n.
            # Assuming num_nodes is a hint, and m, n are primary.
            m = kwargs.get('m')
            n = kwargs.get('n')
            if m is None or n is None:
                raise ValueError("For 'grid_2d', 'm' and 'n' must be provided in kwargs.")
            if m * n != num_nodes:
                logger.warning(f"For 'grid_2d', m*n ({m*n}) does not match num_nodes ({num_nodes}). Using m*n.")
            G = nx.grid_2d_graph(m, n, periodic=kwargs.get('periodic', False))
            num_nodes = m * n # Update num_nodes to actual
            self.set_graph_attrs_data({'num_nodes': num_nodes})
        elif type == 'waxman':
            wm_alpha = kwargs.get('wm_alpha', 0.5)
            wm_beta = kwargs.get('wm_beta', 0.2)
            G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta)
            while not nx.is_connected(G): # Ensure connectivity
                G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta)
            self.set_graph_attrs_data({'wm_alpha': wm_alpha, 'wm_beta': wm_beta})
        elif type == 'random':
            random_prob = kwargs.get('random_prob', 0.5)
            G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
            while not nx.is_connected(G): # Ensure connectivity
                G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
            self.set_graph_attrs_data({'random_prob': random_prob})
        else:
            # This case should not be reached due to the check at the beginning
            raise NotImplementedError(f"Graph type '{type}' generation is not implemented.")

        # Safer way to adopt the structure of G
        self.clear() # Clear current graph data
        self.add_nodes_from(G.nodes(data=True))
        self.add_edges_from(G.edges(data=True))
        # Graph-level attributes from G are not automatically copied by add_nodes/edges.
        # self.graph.update(G.graph) # If you need to copy G's graph attributes.
                                     # Be cautious as it might overwrite BaseNetwork's settings.
                                     # Current design sets num_nodes and type via set_graph_attrs_data.

    def generate_attrs_data(self, node=True, link=True):
        """Generate the data of network attributes based on attributes."""
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
    @cached_property
    def num_nodes(self):
        """Get the number of nodes."""
        return self.number_of_nodes()
    
    @cached_property
    def num_links(self):
        """Get the number of links."""
        return self.number_of_edges()

    @cached_property
    def num_edges(self):
        """Get the number of links."""
        return self.number_of_edges()

    @cached_property
    def links(self):
        """Get the number of links."""
        return EdgeView(self)

    @property
    def adjacency_matrix(self):
        """Get the adjacency matrix of Network."""
        return nx.to_scipy_sparse_matrix(self, format='csr')

    ### Attributes ###
    def get_graph_attrs(self, names):
        """
        Get the attributes of the network.

        Args:
            names (str): The names of the attributes to retrieve. If None, return all attributes.

        Returns:
            dict: A dictionary of network attributes.
        """
        if names is None: return self.graph
        return {attr: self.graph[attr] for attr in names}

    def get_node_attrs(self, types: Optional[Iterable[str]] = None, names: Optional[Iterable[str]] = None) -> List[Any]:
        """
        Get the node attributes of the network.

        Args:
            types: Iterable of types of the node attributes to retrieve. If None, not filtered by type.
            names: Iterable of names of the node attributes to retrieve. If None, not filtered by name.

        Returns:
            A list of node attribute objects.
        """
        if types is None and names is None:
            return list(self.node_attrs.values())

        selected_node_attrs = []
        type_set = set(types) if types is not None else None
        name_set = set(names) if names is not None else None

        for n_attr in self.node_attrs.values():
            match = True
            if type_set is not None and n_attr.type not in type_set:
                match = False
            if name_set is not None and n_attr.name not in name_set:
                match = False
            if match:
                selected_node_attrs.append(n_attr)
        return selected_node_attrs

    def get_link_attrs(self, types: Optional[Iterable[str]] = None, names: Optional[Iterable[str]] = None) -> List[Any]:
        """Get the link attributes of the network.

        Args:
            types: Iterable of types of the link attributes to retrieve. If None, not filtered by type.
            names: Iterable of names of the link attributes to retrieve. If None, not filtered by name.

        Returns:
            A list of link attribute objects.
        """
        if types is None and names is None:
            return list(self.link_attrs.values())

        selected_link_attrs = []
        type_set = set(types) if types is not None else None
        name_set = set(names) if names is not None else None

        for l_attr in self.link_attrs.values():
            match = True
            if type_set is not None and l_attr.type not in type_set:
                match = False
            if name_set is not None and l_attr.name not in name_set:
                match = False
            if match:
                selected_link_attrs.append(l_attr)
        return selected_link_attrs

    @lru_cache
    def get_graph_constraint_attrs(self):
        """Get the constrained graph attributes."""
        # TODO: implement this method
        raise NotImplementedError
    
    @lru_cache
    def get_node_constraint_attrs(self):
        """Get the constrained node attributes."""
        return [n_attr for n_attr in self.node_attrs.values() if n_attr.is_constraint == 'constrained']

    @lru_cache
    def get_link_constraint_attrs(self):
        """Get the constrained link attributes."""
        return [l_attr for l_attr in self.link_attrs.values() if l_attr.is_constraint == 'constrained']

    @property
    def num_node_features(self) -> int:
        """
        Get the number of node features.

        Returns:
            int: The number of node features.
        """
        return len(self.node_attrs)

    @property
    def num_link_features(self) -> int:
        """
        Get the number of link features.

        Returns:
            int: The number of link features.
        """
        return len(self.link_attrs)

    @property
    def num_node_resource_features(self) -> int:
        """
        Get the number of node resource features.

        Returns:
            int: The number of node resource features.
        """
        return len([node_attr for node_attr in self.node_attrs.values() if node_attr.type == 'resource'])

    @property
    def num_link_resource_features(self) -> int:
        """
        Get the number of link resource features.

        Returns:
            int: The number of link resource features.
        """
        return len([link_attr for link_attr in self.link_attrs.values() if link_attr.type == 'resource'])

    ### Set Data ### 
    def init_graph_attrs(self):
        """Initialize the graph attributes storage and copy them to instance attributes."""
        # Ensure default keys for attribute settings exist in self.graph
        if 'node_attrs_setting' not in self.graph:
            self.graph['node_attrs_setting'] = []
        if 'link_attrs_setting' not in self.graph:
            self.graph['link_attrs_setting'] = []
        
        # Make graph attributes (from self.graph) also accessible as instance attributes (self.attr_name)
        # This is done after super().__init__ and after potentially loading incoming_graph_data.graph
        # Avoid overwriting critical properties like 'num_nodes' if it's managed by @cached_property
        # or other specific methods.
        # The __init__ sequence ensures kwargs are processed *after* this.
        for key, value in self.graph.items():
            if key not in ['num_nodes', 'node_attrs_setting', 'link_attrs_setting']: # Example exclusions
                try:
                    setattr(self, key, value)
                except AttributeError: # Some attributes might not be settable (e.g. properties without setters)
                    logger.debug(f"Could not set instance attribute '{key}' from graph attribute.")

    def set_graph_attribute(self, name, value):
        """Set a graph attribute `attr` to `value`."""
        if name in ['num_nodes']:
            self.graph[name] = value
            return
        self.graph[name] = value
        self[name] = value

    def set_graph_attrs_data(self, attributes_data):
        """Set the data of graph attributes."""
        for key, value in attributes_data.items():
            self.set_graph_attribute(key, value)

    def set_node_attrs_data(self, node_attributes_data):
        """Set the data of node attributes."""
        for n_attr, data in node_attributes_data.items():
            n_attr.set_data(self, data)

    def set_link_attrs_data(self, link_attributes_data):
        """Set the data of link attributes."""
        for l_attr, data in link_attributes_data.items():
            l_attr.set_data(self, data)

    ### Get Data ###
    def get_node_attrs_data(self, node_attrs: List[Union[str, Any]]): # node_attrs is a list of attribute objects or names
        """Get the data of node attributes."""
        if not node_attrs: # Handle empty list
            return []
        
        # Determine if attributes are specified by name (str) or by object
        is_str_mode = all(isinstance(attr, str) for attr in node_attrs)
        is_obj_mode = all(not isinstance(attr, str) for attr in node_attrs) # Crude check, assumes homogeneous list

        if not (is_str_mode or is_obj_mode):
            raise ValueError("node_attrs must be a list of all strings (names) or all attribute objects.")

        if is_str_mode:
            node_attrs_data = [list(nx.get_node_attributes(self, str(n_attr_name)).values()) for n_attr_name in node_attrs]
        else:
            node_attrs_data = [n_attr.get_data(self) for n_attr in node_attrs]
        return node_attrs_data

    def get_link_attrs_data(self, link_attrs: List[Union[str, Any]]): # link_attrs is a list of attribute objects or names
        """Get the data of link attributes."""
        if not link_attrs: # Handle empty list
            return []

        is_str_mode = all(isinstance(attr, str) for attr in link_attrs)
        is_obj_mode = all(not isinstance(attr, str) for attr in link_attrs)

        if not (is_str_mode or is_obj_mode):
            raise ValueError("link_attrs must be a list of all strings (names) or all attribute objects.")

        if is_str_mode:
            link_attrs_data = [list(nx.get_edge_attributes(self, str(l_attr_name)).values()) for l_attr_name in link_attrs]
        else: # Object mode
            link_attrs_data = [l_attr.get_data(self) for l_attr in link_attrs]
        return link_attrs_data

    def get_adjacency_attrs_data(self, link_attrs, normalized=False):
        """Get the data of adjacency attributes."""
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

    def subnetwork(self, nodes):
        return self.subgraph(nodes)

    def get_subgraph_view(self, filter_node=no_filter, filter_edge=no_filter):
        view = nx.subgraph_view(self, filter_node=filter_node, filter_edge=filter_edge)
        view.node_attrs = self.node_attrs
        view.link_attrs = self.link_attrs
        return view

    def get_subnetwork_view(self, filter_node=no_filter, filter_edge=no_filter):
        return self.get_subgraph_view(filter_node, filter_edge)

    ### Update ###
    def update_node_resources(self, node_id, v_net_node, method='+'):
        """Update (increase) the value of node atributes."""
        for n_attr in self.node_attrs.keys():
            if n_attr.type != 'resource':
                continue
            n_attr.update(self.nodes[node_id], v_net_node, method)

    def update_link_resources(self, link_pair, v_net_link, method='+'):
        """Update (increase) the value of link atributes."""
        for l_attr in self.link_attrs:
            if l_attr.type != 'resource':
                continue
            l_attr.update(self.links[link_pair], v_net_link, method)

    def update_path_resources(self, path, v_net_link, method='+'):
        """Update (increase) the value of links atributes of path with the same increments."""
        assert len(path) >= 1
        for l_attr in self.link_attrs:
            l_attr.update_path(self, path, v_net_link, method)

    ### Benchmark ###
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

    @lru_cache
    def get_degree_benchmark(self):
        return max(list(dict(self.degree).values()))

    @lru_cache
    def get_node_attr_benchmarks(self, node_attr_types: list = ['resource', 'extrema']):
        if node_attr_types is None:
            n_attrs = self.get_node_attrs()
            node_attr_types = [n_attr.type for n_attr in n_attrs]
        else:
            n_attrs = self.get_node_attrs(node_attr_types)
        n_attrs = self.get_node_attrs(node_attr_types)
        node_data = np.array(self.get_node_attrs_data(n_attrs), dtype=np.float32)
        node_attr_benchmarks = self._get_attr_benchmarks(node_attr_types, n_attrs, node_data)
        return node_attr_benchmarks

    @lru_cache
    def get_link_attr_benchmarks(self, link_attr_types=['resource', 'extrema']):
        if link_attr_types is None:
            l_attrs = self.get_link_attrs()
            link_attr_types = [l_attr.type for l_attr in l_attrs]
        else:
            l_attrs = self.get_link_attrs(link_attr_types)
        link_data = np.array(self.get_link_attrs_data(l_attrs), dtype=np.float32)
        link_data = np.concatenate([link_data, link_data], axis=1)
        link_attr_benchmarks = self._get_attr_benchmarks(link_attr_types, l_attrs, link_data)
        return link_attr_benchmarks

    @lru_cache
    def get_link_sum_attr_benchmarks(self, link_attr_types=['resource', 'extrema']):
        if link_attr_types is None:
            l_attrs = self.get_link_attrs()
            link_attr_types = [l_attr.type for l_attr in l_attrs]
        else:
            l_attrs = self.get_link_attrs(link_attr_types)
        link_sum_attrs_data = np.array(self.get_aggregation_attrs_data(l_attrs, aggr='sum'), dtype=np.float32)
        link_sum_attr_benchmarks = self._get_attr_benchmarks(link_attr_types, l_attrs, link_sum_attrs_data)
        return link_sum_attr_benchmarks


    ### Internal ###
    def __getitem__(self, key):
        """Gets the data of the attribute key or a node."""
        if isinstance(key, (int, str)) and key in self.nodes: # Check if key is a node ID
            return super().__getitem__(key)
        elif isinstance(key, str): # Try to get as an instance attribute
            if hasattr(self, key):
                return getattr(self, key)
            # Fallback to graph attribute if not an instance attribute
            elif key in self.graph:
                return self.graph[key]
            else:
                raise KeyError(f"Attribute or node '{key}' not found.")
        else:
            raise TypeError(f"Unsupported key type for __getitem__: {type(key)}. Expected node ID or attribute name.")

    def __repr__(self):
        net_info = {
            'num_nodes': self.num_nodes,
            'num_links': self.num_links,
            # 'num_node_features': self.num_node_features,
            # 'num_link_features': self.num_link_features,
            'node_attrs': list(self.node_attrs.keys()),
            'link_attrs': list(self.link_attrs.keys())
        }
        net_info_strings = [f'{k}={v}' for k, v in net_info.items()]
        return f"{self.__class__.__name__}({', '.join(net_info_strings)})"

    def __setitem__(self, key: str, value):
        """Sets the attribute key to value."""
        setattr(self, key, value)

    def clone(self):
        return self.__class__.from_dict({
            k: copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })

    def to_gml(self, fpath):
        nx.write_gml(self, fpath)

    @classmethod
    def from_gml(cls: type[_BaseNetworkT], fpath: str, label: str = 'id') -> _BaseNetworkT:
        """
        Create a Network object from a GML file.

        Args:
            fpath (str): The file path of the GML file.
            label (str): The key used for node labels in GML, default is 'id'.
        """
        gml_net = nx.read_gml(fpath, label=label)
        
        # Standardize node labels to integers if they are not already,
        # as many algorithms expect integer node labels.
        # This might require careful handling if original labels are meaningful and non-integer.
        if not all(isinstance(node, int) for node in gml_net.nodes()):
            logger.info("Converting node labels to integers.")
            gml_net = nx.convert_node_labels_to_integers(gml_net, first_label=0, ordering='default')

        # Extract attribute settings if they are stored in the GML graph attributes
        # The __init__ method will use these if they are present in gml_net.graph
        node_attrs_setting = gml_net.graph.get('node_attrs_setting', [])
        link_attrs_setting = gml_net.graph.get('link_attrs_setting', [])

        # Create instance, passing the loaded graph and its settings
        # Other graph attributes from GML will be handled by __init__ & init_graph_attrs
        net = cls(incoming_graph_data=gml_net, 
                  node_attrs_setting=node_attrs_setting, 
                  link_attrs_setting=link_attrs_setting)
        
        # After __init__, self.node_attrs and self.link_attrs are populated based on settings.
        # check_attrs_existence will then verify if these defined attributes are in the actual node/link data.
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