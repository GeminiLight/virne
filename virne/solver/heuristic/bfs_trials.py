# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import networkx as nx

from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.solver.base_solver import Solver, SolverRegistry

from ..base_solver import Solver
from ..rank.node_rank import RWNodeRank, OrderNodeRank, RandomNodeRank


def _ranked_component_bfs_order(network, node_ranking):
    """Order every connected component without changing connected-graph behavior."""
    ordered_nodes = []
    visited_nodes = set()
    for root_node_id in node_ranking:
        if root_node_id in visited_nodes:
            continue
        node_levels = nx.single_source_shortest_path_length(network, root_node_id)
        component_nodes = [
            {
                'node': node_id,
                'level': level,
                'rank': node_ranking[node_id],
            }
            for node_id, level in node_levels.items()
        ]
        component_nodes.sort(key=lambda item: (item['level'], -item['rank']))
        ordered_nodes.extend(item['node'] for item in component_nodes)
        visited_nodes.update(item['node'] for item in component_nodes)
    return ordered_nodes


class BfsSolver(Solver):
    
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs) -> None:
        super(BfsSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.max_visit = kwargs.get('max_visit', config.solver.get('max_visit', 50))
        self.max_depth = kwargs.get('max_depth', config.solver.get('max_depth', 5))
        # ranking strategy
        self.reusable = kwargs.get('reusable', self.reusable)
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', self.matching_mathod)
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', self.shortest_method)
        self.k_shortest = kwargs.get('k_shortest', self.k_shortest)

    def solve(self, instance: dict) -> Solution:
        raise NotImplementedError


@SolverRegistry.register(solver_name='order_rank_bfs', solver_type='heuristic')
class OrderRankBfsSolver(BfsSolver):
    """
    A BFS-based Node Rank solver that ranks nodes by their order in the graph.
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs) -> None:
        super(OrderRankBfsSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.node_rank = OrderNodeRank()

    def solve(self, instance: dict) -> Solution:
        v_net, p_net  = instance['v_net'], instance['p_net']
        v_net_rank = self.node_rank.rank(v_net)
        p_net_rank = self.node_rank.rank(p_net)

        sorted_v_nodes = list(v_net_rank)
        sorted_p_nodes = list(p_net_rank)

        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(
            v_net,
            p_net,
            sorted_v_nodes,
            p_net_init_node,
            self.max_visit,
            self.max_depth,
            shortest_method=self.shortest_method,
            k=self.k_shortest,
        )
        return solution


@SolverRegistry.register(solver_name='random_rank_bfs', solver_type='heuristic')
class RandomRankBfsSolver(BfsSolver):
    """
    A BFS-based Node Rank solver that ranks nodes randomly.
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs) -> None:
        super(RandomRankBfsSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.node_rank = RandomNodeRank()

    def solve(self, instance: dict) -> Solution:
        v_net, p_net  = instance['v_net'], instance['p_net']

        v_net_rank = self.node_rank.rank(v_net)
        p_net_rank = self.node_rank.rank(p_net)

        sorted_v_nodes = list(v_net_rank)
        sorted_p_nodes = list(p_net_rank)

        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(v_net, p_net, sorted_v_nodes, p_net_init_node, 
                                              self.max_visit, self.max_depth,
                                              shortest_method=self.shortest_method,
                                              k=self.k_shortest)
        return solution


@SolverRegistry.register(solver_name='rw_rank_bfs', solver_type='heuristic')
class RandomWalkRankBfsSolver(BfsSolver):
    """
    A BFS-based Node Rank solver that ranks nodes with random walk algorithm.

    References:
        - Cheng et al. "Virtual Network Embedding Through Topology-Aware Node Ranking". In SIGCOMM, 2011.
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs) -> None:
        super(RandomWalkRankBfsSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.node_rank = RWNodeRank()

    def solve(self, instance: dict) -> Solution:
        v_net, p_net  = instance['v_net'], instance['p_net']
        
        v_net_nodes_rank = self.node_rank.rank(v_net)
        p_net_nodes_rank = self.node_rank.rank(p_net)
        sorted_v_nodes = _ranked_component_bfs_order(v_net, v_net_nodes_rank)
        sorted_p_nodes = [n for n in p_net_nodes_rank]
        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(v_net, p_net, sorted_v_nodes, p_net_init_node, 
                                              self.max_visit, self.max_depth,
                                              shortest_method=self.shortest_method,
                                              k=self.k_shortest)
        return solution
