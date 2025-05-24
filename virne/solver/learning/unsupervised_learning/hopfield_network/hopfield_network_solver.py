# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import numpy as np
import networkx as nx

from virne.core import Solution
from virne.core.environment import SolutionStepEnvironment
from .hopfield_network import HopfieldNetwork
from virne.solver.base_solver import Solver, SolverRegistry
from virne.solver.rank.node_rank import GRCNodeRank
from virne.solver.heuristic.node_rank import GRCRankSolver


@SolverRegistry.register(solver_name='hopfield_network', solver_type='u_learning')
class HopfieldNetworkSolver(Solver):
    """
    An Unsupervised Learning-based solver that uses Hopfield Network to construct the subgraph.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(HopfieldNetworkSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.k = 2.5
        self.alpha = 10
        self.beta = 7.0
        self.config.rl.gamma = 3.0
        self.grc_rank = GRCNodeRank()
        self.sub_solver = GRCRankSolver(reusable=self.reusable, verbose=self.verbose)

    def rank_node(self, network):
        n_attrs = network.get_node_attrs(['resource'])
        node_attrs_data = np.array(network.get_node_attrs_data(n_attrs)).T
        node_attr_benchmarks = np.expand_dims(node_attrs_data.max(axis=0), axis=0)
        norm_node_attrs_data = node_attrs_data / node_attr_benchmarks
        node_rank_vector = self.beta * norm_node_attrs_data.mean(axis=-1)
        return node_rank_vector

    def rank_edge(self, network):
        e_attrs = network.get_link_attrs(['resource'])
        adjacency_attrs_data = network.get_adjacency_attrs_data(e_attrs)
        adjacency_attrs_data = np.array([adjacency_attrs_data[i].toarray() for i in range(len(adjacency_attrs_data))])
        adjacency_attr_benchmarks = np.expand_dims(adjacency_attrs_data.max(axis=-1).max(axis=-1), axis=[1, 2])
        # edge_wights = np.divide(adjacency_attr_benchmarks - adjacency_attrs_data, adjacency_attr_benchmarks)
        edge_wights = adjacency_attr_benchmarks - np.divide(adjacency_attrs_data, adjacency_attr_benchmarks)

        edge_wights = edge_wights.mean(axis=0)  # (num_nodes, num_nodes)

        G = nx.Graph(edge_wights)
        distances = nx.floyd_warshall(G)
        distance_max = 0
        for i in range(network.num_nodes):
            for j in range(i, network.num_nodes):
                if distances[i][j] > distance_max:
                    distance_max = distances[i][j]

        link_rank_matrix = np.ones((network.num_nodes, network.num_nodes))
        for i in range(network.num_nodes):
            for j in range(i, network.num_nodes):
                link_rank_matrix[i,j] = self.config.rl.gamma * distances[i][j] / distance_max
                link_rank_matrix[j,i] = self.config.rl.gamma * distances[j][i] / distance_max
        return link_rank_matrix

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        # Enter Event
        chi = self.rank_node(p_net)
        psi = self.rank_edge(p_net)
        num_preselect_nodes = self.number_of_node_selection(v_net)
        zeta = num_preselect_nodes if num_preselect_nodes <= p_net.num_nodes else p_net.num_nodes
        
        h_net = HopfieldNetwork(chi, psi, zeta)
        p_net_node_rank_vector = h_net.execute()

        p_net_subgraph_nodes = [i for i, value in enumerate(p_net_node_rank_vector) if value > 0.5]
        solution = self.grc_solve(v_net, p_net, p_net_subgraph_nodes)
        # p_net_subgraph_nodes = [i for i, value in enumerate(p_net_node_rank_vector) if i <= zeta]

        # p_net_subgraph = p_net.subgraph(p_net_subgraph_nodes)
        # p_net_copy = copy.deepcopy(p_net)
        # n_attrs = p_net.get_node_attrs(['resource'])
        # for p_node_id in p_net.nodes:
        #     if p_node_id in p_net_subgraph_nodes:
        #         continue
        #     for n_attr in n_attrs:
        #         p_net.nodes[p_node_id][n_attr.name] = 0

        # solution = self.sub_solver.solve(v_net, p_net)

        # n_attrs = p_net.get_node_attrs(['resource'])
        # for p_node_id in p_net.nodes:
        #     if p_node_id in p_net_subgraph_nodes:
        #         continue
        #     for n_attr in n_attrs:
        #         p_net.nodes[p_node_id][n_attr.name] = p_net_copy.nodes[p_node_id][n_attr.name]
        
        return solution

    def grc_solve(self, v_net, p_net, p_net_subgraph_nodes):
        def node_mapping(v_net, p_net):
            """Attempt to accommodate VNF in appropriate physical node."""
            v_net_rank = self.grc_rank(v_net)
            p_net_rank = self.grc_rank(p_net)
            sorted_v_nodes = list(v_net_rank)
            sorted_p_nodes = list(p_net_rank)
            sorted_p_nodes = list(set(sorted_p_nodes).intersection(set(p_net_subgraph_nodes)))
            node_mapping_result = self.controller.node_mapper.node_mapping(v_net, p_net, 
                                                            sorted_v_nodes=sorted_v_nodes, 
                                                            sorted_p_nodes=sorted_p_nodes, 
                                                            solution=self.solution, 
                                                            reusable=False, inplace=True, matching_mathod=self.matching_mathod)
            return node_mapping_result

        def link_mapping(v_net, p_net):
            """Seek a path connecting """
            if self.link_rank is None:
                sorted_v_links = v_net.links
            else:
                v_net_edges_rank_dict = self.link_rank(v_net)
                v_net_edges_sort = sorted(v_net_edges_rank_dict.items(), reverse=True, key=lambda x: x[1])
                sorted_v_links = [edge_value[0] for edge_value in v_net_edges_sort]

            link_mapping_result = self.controller.link_mapper.link_mapping(v_net, p_net, solution=self.solution, 
                                                            sorted_v_links=sorted_v_links, 
                                                            shortest_method=self.shortest_method,
                                                            k=self.k_shortest, inplace=True)
            return link_mapping_result
        self.solution = Solution.from_v_net(v_net)
        node_mapping_result = node_mapping(v_net, p_net)
        if node_mapping_result:
            link_mapping_result = link_mapping(v_net, p_net)
            if link_mapping_result:
                # SUCCESS
                self.solution['result'] = True
                return self.solution
            else:
                # FAILURE
                self.solution['route_result'] = False
        else:
            # FAILURE
            self.solution['place_result'] = False
        self.solution['result'] = False
        return self.solution


    def number_of_node_selection(self, v_net):
        return int(self.k * v_net.num_nodes)
    # def hopfield_network(self, chi, psi, zeta):
    #     alpha = 10
    #     net_weights = np.ones((len(chi), len(chi)))
    #     for i in range(zeta):
    #         net_weights[i,i] = 0
    #     i_weight = np.ones(len(chi)) * -(2*zeta-1)
    #     T = -2 * (psi + alpha * net_weights)
    #     I = -(chi + alpha * i_weight)
    #     return (T, I)