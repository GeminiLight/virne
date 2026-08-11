# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import random

from virne.core import Solution
from virne.solver.base_solver import Solver, SolverRegistry
from ..rank.node_rank import FFDNodeRank


class BaseJointPRSolver(Solver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(BaseJointPRSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']

        solution = Solution.from_v_net(v_net)
        for v_node_id in list(v_net.nodes):
            selected_p_net_nodes = list(solution['node_slots'].values())
            candidate_p_net_nodes = self.controller.find_candidate_nodes(v_net, p_net, v_node_id, filter=selected_p_net_nodes)
            if len(candidate_p_net_nodes) == 0:
                # Failure
                solution['place_result'] = False
                self._rollback(p_net, solution)
                return solution
            p_node_id = self.select_p_net_node(p_net, candidate_p_net_nodes)
            place_and_route_result, _ = self.controller.place_and_route(
                v_net,
                p_net,
                v_node_id,
                p_node_id,
                solution,
                shortest_method=self.shortest_method,
                k=self.k_shortest,
            )
            if not place_and_route_result:
                # Failure
                self._rollback(p_net, solution)
                return solution
        # Success
        solution['result'] = True
        return solution

    def _rollback(self, p_net, solution):
        """Restore all resources consumed by the current mapping attempt."""
        for v_link in reversed(list(solution['link_paths'])):
            self.controller.link_mapper.undo_route(v_link, p_net, solution)
        for v_node_id in reversed(list(solution['node_slots'])):
            self.controller.node_mapper.undo_place(v_node_id, p_net, solution)

    def select_p_net_node(self, p_net, candidate_p_net_nodes):
        raise NotImplementedError


@SolverRegistry.register(solver_name='random_joint_pr', solver_type='heuristic')
class RandomJointPRSolver(BaseJointPRSolver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(RandomJointPRSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)

    def select_p_net_node(self, p_net, candidate_p_net_nodes):
        assert len(candidate_p_net_nodes) > 0
        return random.choice(candidate_p_net_nodes)


@SolverRegistry.register(solver_name='order_joint_pr', solver_type='heuristic')
class OrderJointPRSolver(BaseJointPRSolver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(OrderJointPRSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)

    def select_p_net_node(self, p_net, candidate_p_net_nodes):
        assert len(candidate_p_net_nodes) > 0
        return candidate_p_net_nodes[0]
    

@SolverRegistry.register(solver_name='ffd_joint_pr', solver_type='heuristic')
class FFDJointPRSolver(BaseJointPRSolver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(FFDJointPRSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.node_rank = FFDNodeRank()

    def select_p_net_node(self, p_net, candidate_p_net_nodes):
        assert len(candidate_p_net_nodes) > 0
        node_rank_dict = self.node_rank(p_net)
        sorted_p_node = list(node_rank_dict)

        for p_net_node in sorted_p_node:
            if p_net_node in candidate_p_net_nodes:
                return p_net_node
        raise RuntimeError('No ranked physical node found in candidate nodes')
