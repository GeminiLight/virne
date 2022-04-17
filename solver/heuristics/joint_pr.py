from copy import Error
import random
from abc import abstractclassmethod

from base import Controller, Solution
from ..solver import Solver
from ..rank.node_rank import FFDNodeRank


class JointPRSolver(Solver):

    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        super(JointPRSolver, self).__init__(name=name, reusable=reusable, verbose=verbose, **kwargs)

    def solve(self, instance):
        vn, pn = vn, pn  = instance['vn'], instance['pn']

        solution = Solution(vn)
        for vnf_id in list(vn.nodes):
            selected_pn_nodes = list(solution['node_slots'].values())
            candidate_pn_nodes = Controller.find_candidate_nodes(pn, vn, vnf_id, filter=selected_pn_nodes)
            if len(candidate_pn_nodes) == 0:
                # Failure
                solution['place_result'] = False
                return solution
            pn_node_id = self.select_pn_node(pn, candidate_pn_nodes)
            place_and_route_result = Controller.place_and_route(vn, pn, vnf_id, pn_node_id, solution, 
                                                shortest_method='bfs_shortest', k=1)
            if not place_and_route_result:
                # Failure
                solution['route_result'] = False
                return solution
        # Success
        solution['result'] = True
        return solution

    @abstractclassmethod
    def select_pn_node(self, pn, candidate_pn_nodes):
        return NotImplementedError


class RandomJointPRSolver(JointPRSolver):

    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(RandomJointPRSolver, self).__init__(name='random_joint_pr', reusable=reusable, verbose=verbose, **kwargs)

    def select_pn_node(self, pn, candidate_pn_nodes):
        assert len(candidate_pn_nodes) > 0
        return random.choice(candidate_pn_nodes)


class OrderJointPRSolver(JointPRSolver):

    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(OrderJointPRSolver, self).__init__(name='random_joint_pr', reusable=reusable, verbose=verbose, **kwargs)

    def select_pn_node(self, pn, candidate_pn_nodes):
        assert len(candidate_pn_nodes) > 0
        return candidate_pn_nodes[0]
    

class FFDJointPRSolver(JointPRSolver):

    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(FFDJointPRSolver, self).__init__(name='ffd_joint_pr', reusable=reusable, verbose=verbose, **kwargs)
        self.node_rank = FFDNodeRank()

    def select_pn_node(self, pn, candidate_pn_nodes):
        assert len(candidate_pn_nodes) > 0
        node_rank_dict = self.node_rank(pn)
        sorted_p_node = list(node_rank_dict)

        for pn_node in sorted_p_node:
            if pn_node in candidate_pn_nodes:
                return pn_node
        raise Error