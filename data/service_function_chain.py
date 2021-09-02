import networkx as nx
import numpy as np

import os
import sys
file_path_dir = os.path.abspath('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(file_path_dir)

from config import get_config
from data.network import Network

class ServiceFunctionChain(Network):
    def __init__(self, incoming_graph_data=None, num_nodes=None, **kwargs):
        super(ServiceFunctionChain, self).__init__(incoming_graph_data, **kwargs)
        if num_nodes is not None:
            # intialize the topylogy of SFC with the path graph structure.
            self.generate_topology(num_nodes, type='path')

    @property
    def length(self):
        r"""Return the length of SFC."""
        return self.num_nodes

    @property
    def sum_requests(self):
        r"""Return the sum of all SFC requests."""
        nodes_data = np.array(self.get_nodes_data(self.node_attrs))
        edges_data = np.array(self.get_edges_data(self.edge_attrs))
        return nodes_data.sum(axis=(0,1)) + edges_data.sum(axis=(0,1))

    @property
    def raw_state(self):
        nodes_data = np.array(self.get_nodes_data(self.node_attrs))
        edges_data = np.array(self.get_edges_data(self.edge_attrs))
        edges_data = np.insert(edges_data, 0, 0, axis=1)
        return np.vstack([nodes_data, edges_data])

    # def get_node_requests(self, nid):
    #     return {self.curr_sfc.nodes[nid][n_attr] for n_attr in self.node_attrs}

    # def get_edge_requests(self, eid):
    #     return {self.curr_sfc.edges[eid][e_attr] for e_attr in self.edge_attrs}

if __name__ == '__main__':
    config, _ = get_config()

    # sfc = ServiceFunctionChain(num_nodes=10, id=1)
    # sfc.generate_data(['cpu'], ['bw'], 0, 20, 0, 20)
    # sfc.to_csv()
    # new_sfc = ServiceFunctionChain.from_csv("./dataset/node_data.csv", "./dataset/edge_data.csv")