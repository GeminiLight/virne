import copy
import random

from base.controller import Controller
from base.recorder import Solution, Counter


class State:
    """
    The state of MCTS to record the state in one node.
    game score, round count, record
    """
    def __init__(self, pn, vn):
        self.pn = copy.deepcopy(pn)
        self.vn = vn
        self.vn_node_id = -1  # wait
        self.pn_node_id = pn.num_nodes
        self.selected_pn_nodes = []
        self.max_expansion = pn.num_nodes

    def is_terminal(self):
        """Check the state is a terminal state"""
        # Case 1: current virtual node is the lastest one
        # Case 2: there is no node which is capable of accommodate the current virtual node
        if self.vn_node_id == self.vn.num_nodes - 1 or self.pn_node_id == -1:
            return True
        else:
            return False

    def compute_final_reward(self):
        """
        Success: reward = revenue - cost
        Failure: reward = -inf
        """
        solution = Solution(self.vn)
        for i in range(self.vn.num_nodes):
            solution['node_slots'].update({i: self.selected_pn_nodes[i]})
        link_result = Controller.link_mapping(self.vn, self.pn, solution=solution, available_network=True,
                                                shortest_method='bfs_shortest', k=1, inplace=True)

        if link_result:
            vn_cost = Counter.calculate_vn_cost(self.vn, solution)
            vn_revenue = Counter.calculate_vn_revenue(self.vn)
            return 1000 + vn_revenue - vn_cost
        else:
            return -float('inf')

    def random_select_next_state(self):
        """Random select a physical node to accommodate the next virtual node"""
        candidate_pn_nodes = Controller.find_candidate_nodes(
            pn=self.pn, 
            vn=self.vn, 
            vn_node_id=self.vn_node_id+1, 
            filter=self.selected_pn_nodes)

        self.max_expansion = len(candidate_pn_nodes)
        if self.max_expansion > 0:
            random_choice = random.choice([action for action in candidate_pn_nodes])
        else:
            random_choice = -1
        next_state = copy.deepcopy(self)
        next_state.vn_node_id = self.vn_node_id + 1
        next_state.pn_node_id = random_choice
        next_state.selected_pn_nodes = self.selected_pn_nodes + [random_choice]
        return next_state


class Node:
    """
    Node of Monte Carlo Tree
    """
    def __init__(self, parent=None, state=None):
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = []
        self.state = state
        self.visit_times = 0
        self.value = 0.0

    def is_complete_expand(self):
        if len(self.children) == self.state.max_expansion:
            return True
        else:
            return False