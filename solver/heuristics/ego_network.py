import numpy as np
import networkx as nx

from base import Controller, Solution
from ..solver import Solver
from ..rank.node_rank import OrderNodeRank, GRCNodeRank, FFDNodeRank, NRMNodeRank, RWNodeRank, RandomNodeRank
from ..rank.edge_rank import OrderEdgeRank, FFDEdgeRank


class EgoNetworkSolver(Solver):
    
    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(EgoNetworkSolver, self).__init__(name='ego_vne', reusable=reusable, verbose=verbose, **kwargs)
        self.node_rank = NRMNodeRank()
        self.edge_rank = None
        self.beta = 0.3 # 0.9

    def solve(self, instance):
        vn, pn  = instance['vn'], instance['pn']
        self.solution = Solution(vn)
        mapping_result = self.mapping(vn, pn)
        self.solution['result'] = True if mapping_result else False
        return self.solution

    def mapping(self, vn, pn):
        r"""Attempt to accommodate VNF in appropriate physical node."""
        vn_node_rank = self.node_rank(vn)

        num_v_nodes = 0
        for v_node_id in list(vn_node_rank.keys()):
            if num_v_nodes == 0:
                candidate_nodes = self.controller.find_candidate_nodes(pn, vn, v_node_id)
                pn_node_rank = self.calc_pn_node_rank(pn)

        
        node_mapping_result = Controller.node_mapping(vn, pn, 
                                                        sorted_v_nodes=sorted_v_nodes, 
                                                        sorted_p_nodes=sorted_p_nodes, 
                                                        solution=self.solution, 
                                                        reusable=False, inplace=True)
        return node_mapping_result

    def link_mapping(self, vn, pn):
        r"""Seek a path connecting """
        if self.edge_rank is None:
            sorted_v_edges = vn.edges
        else:
            vn_edges_rank_dict = self.edge_rank(vn)
            vn_edges_sort = sorted(vn_edges_rank_dict.items(), reverse=True, key=lambda x: x[1])
            sorted_v_edges = [edge_value[0] for edge_value in vn_edges_sort]

        link_mapping_result = Controller.link_mapping(vn, pn, solution=self.solution, 
                                                        sorted_v_edges=sorted_v_edges, 
                                                        shortest_method='bfs_shortest', 
                                                        k=1, inplace=True)
        return link_mapping_result

    def calc_pn_node_rank(self, network):
        free_nodes_resources = network.get_node_attrs_data(network.get_node_attrs('resource'))
        free_nodes_resources = np.array(free_nodes_resources, dtype=np.float32).sum(axis=0)

        free_edge_resources = network.get_aggregation_attrs_data(network.get_edge_attrs('resource'))
        free_edge_resources = np.array(free_edge_resources, dtype=np.float32).sum(axis=0)

        node_degrees = nx.degree(network)

        node_rank = self.beta * free_nodes_resources + (1.0 - self.beta) * node_degrees * free_edge_resources
        return node_rank