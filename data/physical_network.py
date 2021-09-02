import networkx as nx
import numpy as np

import os
import sys
file_path_dir = os.path.abspath('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(file_path_dir)

from config import get_config

from data.network import Network
from data.utils import get_items

class PhysicalNetwork(Network):
    def __init__(self, incoming_graph_data=None, num_nodes=None, **kwargs):
        super(PhysicalNetwork, self).__init__(incoming_graph_data, **kwargs)
        # Initialize the topology
        if num_nodes is not None:
            wm_alpha = kwargs.get('wm_alpha', 0.5)
            wm_beta = kwargs.get('wm_beta', 0.2)
            self.generate_topology(num_nodes, type='waxman', wm_alpha=wm_alpha, wm_beta=wm_beta)
        self.original_edge_attrs = []
        self.original_node_attrs = []
        self.extended_node_attrs = []
        self.invariant_node_attrs = []

    def generate_data(self, node_attrs=[], edge_attrs=[],
                        min_node_value=2, max_node_value=30, 
                        min_edge_value=2, max_edge_value=30):
        super().generate_data(node_attrs, edge_attrs,
                        min_node_value, max_node_value, 
                        min_edge_value, max_edge_value)
        self.original_node_attrs = node_attrs
        self.original_edge_attrs = edge_attrs
        for n_attr in node_attrs:
            self.set_node_attr('max_' + n_attr, self.get_node_attr(n_attr))
        # convert edge attributes to node attributes
        for e_attr in edge_attrs:
            self.set_node_attr('sum_' + e_attr, self.num_nodes*[0])
            for n in self.nodes:
                bw_sum = 0
                for neighbor in self.adj[n]:
                    bw_sum += self.edges[n, neighbor]['bw']
                self.nodes[n]['sum_' + e_attr] = self.nodes[n]['max_sum_' + e_attr] = bw_sum
                    
        self.extended_node_attrs = ['sum_' + e_attr for e_attr in self.original_edge_attrs]
        self.invariant_node_attrs =  ['max_' + n_attr for n_attr in self.original_node_attrs + self.extended_node_attrs]

    def draw(self, show=True, save_path=None, **kwargs):
        from matplotlib import pyplot as plt
        from networkx.drawing.nx_pydot import graphviz_layout

        # nx.draw(artist_subtree, pos=pos,
        #         node_size=[(v+3)*50 for v in nodes_size], alpha=0.5, 
        #         node_color=list(dict(spl).values()),
        #         cmap=plt.cm.Blues_r)
        plt.figure(figsize=(12, 8), dpi=200)
        edge_colors = self.get_edge_attr('bw')
        nx.draw(self, 
                node_size=10, 
                node_color='#fa9d4b',
                with_label=True, 
                edge_color=edge_colors,
                edge_cmap=plt.cm.Blues, 
                alpha=0.5, )
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    @classmethod
    def from_csv(cls, node_path, edge_path):
        net = super().from_csv(node_path, edge_path)
        all_node_attrs = net.node_attrs
        for attr in net.node_attrs:
            if attr.startswith('sum'):
                net.extended_node_attrs.append(attr)
            elif attr.startswith('max'):
                net.invariant_node_attrs.append(attr)
            elif attr == 'pos':
                continue
            else:
                net.original_node_attrs.append(attr)
        return net

    def find_candidate_nodes(self, attrs=None, req_values=[], filter=[], rtype='id'):
        r"""Find candicate nodes according to the restrictions and filter.

        Args:
            attrs (list or dict)
            req_values (list)
        
        Returns:
            candicate_nodes (`rtype`)
        """
        items = get_items(attrs, req_values)
        if rtype == 'id':
            candidate_nodes = list(self.nodes)
            for name, req in items:
                suitable_nodes = np.where(self.get_node_attr(name, rtype='array') >= req)
                candidate_nodes = np.intersect1d(candidate_nodes, suitable_nodes)
            candidate_nodes = np.setdiff1d(candidate_nodes, filter)
        elif rtype == 'bool':
            candidate_nodes = np.ones(self.num_nodes)
            for  name, req in items:
                suitable_nodes = np.where(self.get_node_attr(name, rtype='array') >= req, True, False)
                candidate_nodes = np.logical_and(candidate_nodes, suitable_nodes)
            candidate_nodes[filter] = False
        return candidate_nodes

    def is_candidate_node(self, nid, attrs=None, req_values=[]):
        r"""Return `True` if `nid` is candidate node  or `False`."""
        items = get_items(attrs, req_values)
        for name, req in items:
            if self.nodes[nid][name] < req:
                return False
        return True

    @property
    def raw_state(self):
        nodes_data = self.get_nodes_data(self.original_node_attrs + self.extended_node_attrs)
        return nodes_data

    @property
    def norm_benchmark(self):
        invariant_node_data = self.get_nodes_data(self.invariant_node_attrs)
        benchmark = np.array(invariant_node_data).max(axis=1)
        return benchmark


if __name__ == '__main__':
    config, _ = get_config()
    # generate pn dataset
    # pn = PhysicalNetwork(num_nodes=config.pn_num_nodes, 
    #                      wm_alpha=config.wm_alpha,
    #                      wm_beta=config.wm_beta)
    # pn.generate_data(node_attrs=config.node_attrs, 
    #                  edge_attrs=config.edge_attrs, 
    #                  min_node_value=config.min_node_capacity, 
    #                  max_node_value=config.max_node_capacity,
    #                  min_edge_value=config.min_edge_capacity, 
    #                  max_edge_value=config.max_edge_capacity)
    # pn.to_csv(config.pn_node_dataset, config.pn_edge_dataset)
    pn = PhysicalNetwork.from_csv(config.pn_node_dataset, config.pn_edge_dataset)
    pn.draw()
    