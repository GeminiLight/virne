# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import networkx as nx
from abc import abstractclassmethod

from base import Controller, Recorder, Counter, Solution
from base.environment import SolutionStepEnvironment

from ..solver import Solver
from ..rank.node_rank import RWNodeRank, OrderNodeRank, RandomNodeRank
from solver import registry


class BfsSolver(Solver):
    
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, **kwargs) -> None:
        super(BfsSolver, self).__init__(controller, recorder, counter, **kwargs)

    @abstractclassmethod
    def solve(self, instance: dict) -> Solution:
        raise NotImplementedError


@registry.register(
    solver_name='order_rank_bfs', 
    env_cls=SolutionStepEnvironment,
    solver_type='heuristic')
class OrderRankBfsSolver(BfsSolver):
    """
    A BFS-based Node Rank solver that ranks nodes by their order in the graph.
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, **kwargs) -> None:
        super(OrderRankBfsSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = OrderNodeRank()

    def solve(self, instance: dict) -> Solution:
        v_net, p_net  = instance['v_net'], instance['p_net']
        v_net_rank = self.node_rank.rank(v_net)
        p_net_rank = self.node_rank.rank(p_net)

        sorted_v_nodes = list(v_net_rank)
        sorted_p_nodes = list(p_net_rank)

        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(v_net, p_net, sorted_v_nodes, p_net_init_node, shortest_method=self.shortest_method)
        return solution


@registry.register(
    solver_name='random_rank_bfs', 
    env_cls=SolutionStepEnvironment,
    solver_type='heuristic')
class RandomRankBfsSolver(BfsSolver):
    """
    A BFS-based Node Rank solver that ranks nodes randomly.
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, **kwargs) -> None:
        super(RandomRankBfsSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = RandomNodeRank()

    def solve(self, instance: dict) -> Solution:
        v_net, p_net  = instance['v_net'], instance['p_net']

        v_net_rank = self.node_rank.rank(v_net)
        p_net_rank = self.node_rank.rank(p_net)

        sorted_v_nodes = list(v_net_rank)
        sorted_p_nodes = list(p_net_rank)

        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(v_net, p_net, sorted_v_nodes, p_net_init_node)
        return solution

@registry.register(
    solver_name='rw_rank_bfs', 
    env_cls=SolutionStepEnvironment,
    solver_type='heuristic')
class RandomWalkRankBfsSolver(BfsSolver):
    """
    A BFS-based Node Rank solver that ranks nodes with random walk algorithm.

    References:
        - Cheng et al. "Virtual Network Embedding Through Topology-Aware Node Ranking". In SIGCOMM, 2011.
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, **kwargs) -> None:
        super(RandomWalkRankBfsSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = RWNodeRank()

    def solve(self, instance: dict) -> Solution:
        v_net, p_net  = instance['v_net'], instance['p_net']
        
        v_net_nodes_rank = self.node_rank.rank(v_net)
        p_net_nodes_rank = self.node_rank.rank(p_net)
        largest_rank_vid = list(v_net_nodes_rank.keys())[0]
        v_net_node_level = nx.single_source_shortest_path_length(v_net, largest_rank_vid)
        v_net_node_level_rank_list = []
        for node, level in v_net_node_level.items():
            v_net_node_level_rank_list.append({'node': node, 'level': level, 'rank': v_net_nodes_rank[node]})
        sorted_v_net_node = sorted(v_net_node_level_rank_list, key=lambda r: (r['level'], -r['rank']))

        sorted_v_nodes = [n['node'] for n in sorted_v_net_node]
        sorted_p_nodes = [n for n in p_net_nodes_rank]
        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(v_net, p_net, sorted_v_nodes, p_net_init_node)
        return solution
