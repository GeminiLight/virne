# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from copy import Error
import random
from abc import abstractclassmethod

from virne.base import Solution
from ..solver import Solver
from ..rank.node_rank import FFDNodeRank


class JointPRSolver(Solver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(JointPRSolver, self).__init__(controller, recorder, counter, **kwargs)

    def solve(self, instance):
        v_net, p_net = v_net, p_net  = instance['v_net'], instance['p_net']

        solution = Solution(v_net)
        for v_node_id in list(v_net.nodes):
            selected_p_net_nodes = list(solution['node_slots'].values())
            candidate_p_net_nodes = self.controller.find_candidate_nodes(v_net, p_net, v_node_id, filter=selected_p_net_nodes)
            if len(candidate_p_net_nodes) == 0:
                # Failure
                solution['place_result'] = False
                return solution
            p_node_id = self.select_p_net_node(p_net, candidate_p_net_nodes)
            place_and_route_result, place_and_route_info = self.controller.place_and_route(v_net, p_net, v_node_id, p_node_id, solution, 
                                                shortest_method=self.shortest_method, k=1)
            if not place_and_route_result:
                # Failure
                solution['route_result'] = False
                return solution
        # Success
        solution['result'] = True
        return solution

    @abstractclassmethod
    def select_p_net_node(self, p_net, candidate_p_net_nodes):
        return NotImplementedError


class RandomJointPRSolver(JointPRSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(RandomJointPRSolver, self).__init__(controller, recorder, counter, **kwargs)

    def select_p_net_node(self, p_net, candidate_p_net_nodes):
        assert len(candidate_p_net_nodes) > 0
        return random.choice(candidate_p_net_nodes)


class OrderJointPRSolver(JointPRSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(OrderJointPRSolver, self).__init__(controller, recorder, counter, **kwargs)

    def select_p_net_node(self, p_net, candidate_p_net_nodes):
        assert len(candidate_p_net_nodes) > 0
        return candidate_p_net_nodes[0]
    

class FFDJointPRSolver(JointPRSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(FFDJointPRSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = FFDNodeRank()

    def select_p_net_node(self, p_net, candidate_p_net_nodes):
        assert len(candidate_p_net_nodes) > 0
        node_rank_dict = self.node_rank(p_net)
        sorted_p_node = list(node_rank_dict)

        for p_net_node in sorted_p_node:
            if p_net_node in candidate_p_net_nodes:
                return p_net_node
        raise Error