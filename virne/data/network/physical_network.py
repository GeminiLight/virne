# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from functools import cached_property
import os
import copy
import random
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional, Iterable, Union, TypeVar # Added TypeVar

import logging
logger = logging.getLogger(__name__)

from .base_network import BaseNetwork
from ..attribute import NodeStatusAttribute, LinkStatusAttribute


class PhysicalNetwork(BaseNetwork):
    """
    PhysicalNetwork class is a subclass of Network class. It represents a physical network.

    Attributes:
        degree_benchmark (dict): The degree benchmark for the network.
        node_attr_benchmarks (dict): The node attribute benchmarks for the network.
        link_attr_benchmarks (dict): The link attribute benchmarks for the network.

    Methods:
        from_topology_zoo_setting(topology_zoo_setting: Dict[str, Any], seed: Optional[int] = None) -> PhysicalNetwork:
            Returns a PhysicalNetwork object generated from the Topology Zoo data, with optional seed for reproducibility.

        save_dataset(self, dataset_dir: str) -> None:
            Saves the dataset as a .gml file in the specified directory.

        load_dataset(dataset_dir: str) -> PhysicalNetwork:
            Loads the dataset from the specified directory as a PhysicalNetwork object, 
            and calculates the benchmarks for normalization.
    """
    def __init__(self, incoming_graph_data: nx.Graph = None, node_attrs_setting: list = [], link_attrs_setting: list = [], **kwargs) -> None:
        """
        Initialize a PhysicalNetwork object.

        Args:
            incoming_graph_data (nx.Graph): An existing graph object (optional).
            node_attrs_setting (list): Node attribute settings (default []).
            link_attrs_setting (list): Link attribute settings (default []).
            **kwargs: Additional keyword arguments.
        """
        super(PhysicalNetwork, self).__init__(incoming_graph_data, node_attrs_setting, link_attrs_setting, **kwargs)

    def generate_topology(self, num_nodes: int, type: str = 'waxman', **kwargs):
        """
        Generate a topology for the network.

        Args:
            num_nodes (int): The number of nodes in the network.
            type (str, optional): The type of network to generate. Defaults to 'waxman'.
            **kwargs: Keyword arguments to pass to the network generator.
        """
        super().generate_topology(num_nodes, type, **kwargs)
        self.degree_benchmark = self.get_degree_benchmark()

    def generate_attrs_data(self, node: bool = True, link: bool = True) -> None:
        """
        Generate attribute data for the network.

        Args:
            node (bool, optional): Whether or not to generate node attribute data. Defaults to True.
            link (bool, optional): Whether or not to generate link attribute data. Defaults to True.
        """
        super().generate_attrs_data(node, link)
        if node: 
            self.node_attr_benchmarks = self.get_node_attr_benchmarks()
        if link:
            self.link_attr_benchmarks = self.get_link_attr_benchmarks()
            self.link_sum_attr_benchmarks = self.get_link_sum_attr_benchmarks()

    @staticmethod
    def _load_pnet_from_gml(gml_file_path: str, 
                            node_attrs_setting: list, 
                            link_attrs_setting: list, 
                            graph_settings: dict, 
                            seed: Optional[int] = None,
                            generate_attributes_after_load: bool = False) -> 'PhysicalNetwork':
        """Helper to load and initialize PhysicalNetwork from GML."""
        
        # Create an initial instance with settings.
        # The actual graph data will be replaced by GML content.
        # We pass graph_settings which might include 'name', 'type', etc.
        # These might be overwritten by GML graph attributes if present.
        p_net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, 
                                link_attrs_setting=link_attrs_setting, 
                                **graph_settings)

        G_from_gml = nx.read_gml(gml_file_path, label='id')

        # Preserve original node/link attr settings, but update graph attributes from GML
        # BaseNetwork.__init__ already handles incoming_graph_data.graph attributes.
        # Here, we explicitly clear and reload to ensure GML structure is primary.
        
        p_net.clear() # Clear any default structure from __init__
        p_net.graph.update(G_from_gml.graph) # Update graph attributes from GML
                                            # This might overwrite some initial graph_settings.
        
        # Re-initialize node_attrs_setting and link_attrs_setting in p_net.graph
        # if they were overwritten by G_from_gml.graph or ensure they are the ones passed in.
        p_net.graph['node_attrs_setting'] = node_attrs_setting
        p_net.graph['link_attrs_setting'] = link_attrs_setting
        
        # Re-create attribute objects based on the (potentially updated) settings
        p_net.create_attrs_from_setting()

        # Add nodes and edges from GML
        p_net.add_nodes_from(G_from_gml.nodes(data=True))
        p_net.add_edges_from(G_from_gml.edges(data=True))

        # Dynamically add Node/LinkStatusAttribute for attributes found in GML
        # but not defined in the initial settings.
        if p_net.nodes:
            first_node_data = p_net.nodes[list(p_net.nodes)[0]]
            for attr_name in first_node_data.keys():
                if attr_name not in p_net.node_attrs:
                    logger.info(f"Dynamically adding NodeStatusAttribute for '{attr_name}' from GML.")
                    p_net.node_attrs[attr_name] = NodeStatusAttribute(attr_name)
        if p_net.edges: # Use p_net.edges
            first_edge_data = p_net.edges[list(p_net.edges)[0]]
            for attr_name in first_edge_data.keys():
                if attr_name not in p_net.link_attrs:
                    logger.info(f"Dynamically adding LinkStatusAttribute for '{attr_name}' from GML.")
                    p_net.link_attrs[attr_name] = LinkStatusAttribute(attr_name)
        
        # check_attrs_existence can be called here to validate
        p_net.check_attrs_existence()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if generate_attributes_after_load:
            # This call implies that some attributes defined in settings
            # need their data (re)generated even after loading from GML.
            p_net.generate_attrs_data(node=True, link=True)
        
        # Set benchmarks (original logic)
        p_net.degree_benchmark = p_net.get_degree_benchmark()
        if p_net.node_attrs: # Check if node_attrs exist before getting benchmarks
             p_net.node_attr_benchmarks = p_net.get_node_attr_benchmarks()
        if p_net.link_attrs: # Check if link_attrs exist
             p_net.link_attr_benchmarks = p_net.get_link_attr_benchmarks()
             p_net.link_sum_attr_benchmarks = p_net.get_link_sum_attr_benchmarks()
        
        logger.info(f"Loaded topology from {gml_file_path}")
        return p_net

    @staticmethod
    def from_setting(setting: dict, seed: int = None) -> 'PhysicalNetwork':
        """
        Create a PhysicalNetwork object from the given setting.
        Can load from a GML file if specified, otherwise generates topology.
        """
        setting = copy.deepcopy(setting)
        node_attrs_setting = setting.pop('node_attrs_setting', [])
        link_attrs_setting = setting.pop('link_attrs_setting', [])
        
        topology_config = setting.pop('topology', {})
        file_path = topology_config.get('file_path')

        # graph_init_settings are remaining items in 'setting' (e.g. name, id)
        # plus specific topology params if not loading file (e.g. type for generation)
        graph_init_settings = setting 

        if file_path and os.path.exists(file_path):
            try:
                # For GML loaded via from_setting, original code did NOT call generate_attrs_data.
                # Benchmarks were set after.
                return PhysicalNetwork._load_pnet_from_gml(
                    gml_file_path=file_path,
                    node_attrs_setting=node_attrs_setting,
                    link_attrs_setting=link_attrs_setting,
                    graph_settings=graph_init_settings,
                    seed=seed,
                    generate_attributes_after_load=False # Consistent with original from_setting
                )
            except Exception as e:
                logger.error(f"Error loading topology from {file_path}: {e}. Proceeding to generate topology.")
                # Fall through to generation if GML loading fails
        
        # Generate topology if file_path is not valid or loading failed
        num_nodes = graph_init_settings.pop('num_nodes') # num_nodes is for generation
        
        # Pass remaining topology_config (like type, wm_alpha) and graph_init_settings
        # to the PhysicalNetwork constructor and generate_topology
        # kwargs for constructor:
        init_kwargs = {**graph_init_settings, **topology_config}
        # topology_type = topology_config.pop('type', 'waxman') # Default type for generation

        p_net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, 
                                link_attrs_setting=link_attrs_setting, 
                                **init_kwargs) # Pass all relevant graph settings

        if seed is None: # Apply seed before generation
            seed = init_kwargs.get('seed') # Check if seed was in settings
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        p_net.generate_topology(num_nodes=num_nodes, **topology_config) # type is in topology_config
        p_net.generate_attrs_data() # Generates data for attributes and sets benchmarks
        return p_net

    @staticmethod
    def from_topology_zoo_setting(topology_zoo_setting: dict, seed: int = None) -> 'PhysicalNetwork':
        """
        Create a PhysicalNetwork object from a topology zoo setting (GML file).
        """
        setting = copy.deepcopy(topology_zoo_setting)
        node_attrs_setting = setting.pop('node_attrs_setting', [])
        link_attrs_setting = setting.pop('link_attrs_setting', [])
        file_path = setting.pop('file_path') # Must exist for topology_zoo

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"GML file_path for topology_zoo_setting not found or not specified: {file_path}")

        # Remaining 'setting' items are graph-level attributes
        graph_settings = setting
        
        # Original from_topology_zoo_setting DID call generate_attrs_data.
        return PhysicalNetwork._load_pnet_from_gml(
            gml_file_path=file_path,
            node_attrs_setting=node_attrs_setting,
            link_attrs_setting=link_attrs_setting,
            graph_settings=graph_settings,
            seed=seed,
            generate_attributes_after_load=True # Consistent with original from_topology_zoo_setting
        )

    def save_dataset(self, dataset_dir: str) -> None:
        """
        Save the physical network dataset to a directory.

        Args:
            dataset_dir (str): The path to the directory where the physical network dataset is to be saved.
        """
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        file_path = os.path.join(dataset_dir, 'p_net.gml')
        self.to_gml(file_path)

    @staticmethod
    def load_dataset(dataset_dir: str) -> 'PhysicalNetwork':
        """
        Load the physical network dataset from a directory.
        Assumes p_net.gml exists and contains all necessary info.
        """
        if not os.path.exists(dataset_dir):
            raise ValueError(f'Find no dataset in {dataset_dir}.\nPlease firstly generating it.')
        file_path = os.path.join(dataset_dir, 'p_net.gml')
        
        # Using BaseNetwork.from_gml and then casting/initializing benchmarks
        # Or, if p_net.gml is saved by PhysicalNetwork and contains all settings,
        # _load_pnet_from_gml could be adapted.
        # For simplicity, let's assume from_gml from BaseNetwork is sufficient,
        # and then we manually set up PhysicalNetwork specific parts.
        
        # This approach re-uses the robust from_gml from BaseNetwork
        # It assumes that node/link_attrs_setting are part of the GML's graph attributes
        # or that the GML structure itself is enough.
        p_net_base = BaseNetwork.from_gml(file_path) # Returns a BaseNetwork instance

        # Convert BaseNetwork to PhysicalNetwork. This is a bit tricky.
        # A cleaner way is if from_gml could instantiate the correct subclass.
        # Or, PhysicalNetwork.from_gml directly.
        # Let's define PhysicalNetwork.from_gml if not already.
        # For now, assuming p_net.gml was saved by PhysicalNetwork and might have specific settings.

        # Read GML to get graph data and graph attributes (which should include settings)
        temp_g = nx.read_gml(file_path, label='id')
        node_attrs_setting = temp_g.graph.get('node_attrs_setting', [])
        link_attrs_setting = temp_g.graph.get('link_attrs_setting', [])
        
        # Remove settings from graph_attrs before passing to _load_pnet_from_gml's graph_settings
        graph_attrs_from_gml = {k: v for k, v in temp_g.graph.items() 
                                if k not in ['node_attrs_setting', 'link_attrs_setting']}

        p_net = PhysicalNetwork._load_pnet_from_gml(
            gml_file_path=file_path,
            node_attrs_setting=node_attrs_setting,
            link_attrs_setting=link_attrs_setting,
            graph_settings=graph_attrs_from_gml,
            seed=None, # Seed not typically applied when loading a finished dataset
            generate_attributes_after_load=False # Data should be in GML
        )
        # Benchmarks are set within _load_pnet_from_gml
        return p_net
