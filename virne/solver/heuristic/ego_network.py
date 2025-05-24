# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import numpy as np
import networkx as nx

from virne.core import Solution
from virne.solver.base_solver import Solver, SolverRegistry
from ..rank.node_rank import OrderNodeRank, GRCNodeRank, FFDNodeRank, NRMNodeRank, RWNodeRank, RandomNodeRank
from ..rank.link_rank import OrderLinkRank, FFDLinkRank


class EgoNetworkSolver(Solver):
    
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(EgoNetworkSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.node_rank = NRMNodeRank()
        self.link_rank = None
        self.beta = 0.3 # 0.9

    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']
        self.solution = Solution.from_v_net(v_net)
        mapping_result = self.mapping(v_net, p_net)
        self.solution['result'] = True if mapping_result else False
        return self.solution

    def mapping(self, v_net, p_net):
        """Attempt to accommodate VNF in appropriate physical node."""
        v_net_node_rank = self.node_rank(v_net)

        num_v_nodes = 0
        for v_node_id in list(v_net_node_rank.keys()):
            if num_v_nodes == 0:
                candidate_nodes = self.controller.find_candidate_nodes(v_net, p_net, v_node_id)
                p_net_node_rank = self.calc_p_net_node_rank(p_net)

        
        node_mapping_result = self.controller.node_mapper.node_mapping(v_net, p_net, 
                                                        sorted_v_nodes=sorted_v_nodes, 
                                                        sorted_p_nodes=sorted_p_nodes, 
                                                        solution=self.solution, 
                                                        reusable=False, inplace=True)
        return node_mapping_result

    def link_mapping(self, v_net, p_net):
        if self.link_rank is None:
            sorted_v_links = v_net.links
        else:
            v_net_edges_rank_dict = self.link_rank(v_net)
            v_net_edges_sort = sorted(v_net_edges_rank_dict.items(), reverse=True, key=lambda x: x[1])
            sorted_v_links = [edge_value[0] for edge_value in v_net_edges_sort]

        link_mapping_result = self.controller.link_mapper.link_mapping(v_net, p_net, solution=self.solution, 
                                                        sorted_v_links=sorted_v_links, 
                                                        shortest_method='bfs_shortest', 
                                                        k=1, inplace=True)
        return link_mapping_result

    def calc_p_net_node_rank(self, network):
        free_nodes_resources = network.get_node_attrs_data(network.get_node_attrs('resource'))
        free_nodes_resources = np.array(free_nodes_resources, dtype=np.float32).sum(axis=0)

        free_edge_resources = network.get_aggregation_attrs_data(network.get_link_attrs('resource'))
        free_edge_resources = np.array(free_edge_resources, dtype=np.float32).sum(axis=0)

        node_degrees = nx.degree(network)

        node_rank = self.beta * free_nodes_resources + (1.0 - self.beta) * node_degrees * free_edge_resources
        return node_rank