# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import random

from virne.core import Solution


class State:
    """
    The state of MCTS to record the state in one node.
    game score, round count, record
    """
    def __init__(self, p_net, v_net, controller, recorder, counter):
        self.p_net = copy.deepcopy(p_net)
        self.v_net = v_net
        self.controller = controller
        self.counter = counter
        self.v_node_id = -1  # wait
        self.p_node_id = p_net.num_nodes
        self.selected_p_net_nodes = []
        self.max_expansion = p_net.num_nodes

    def is_terminal(self):
        """Check the state is a terminal state"""
        # Case 1: current virtual node is the lastest one
        # Case 2: there is no node which is capable of accommodate the current virtual node
        if self.v_node_id == self.v_net.num_nodes - 1 or self.p_node_id == -1:
            return True
        else:
            return False

    def compute_final_reward(self):
        """
        Success: reward = revenue - cost
        Failure: reward = -inf
        """
        solution = Solution.from_v_net(self.v_net)
        for i in range(self.v_net.num_nodes):
            solution['node_slots'].update({i: self.selected_p_net_nodes[i]})
        link_result = self.controller.link_mapper.link_mapping(self.v_net, self.p_net, solution=solution,
                                                shortest_method='bfs_shortest', k=1, inplace=True)

        if link_result:
            v_net_cost = self.counter.calculate_v_net_cost(self.v_net, solution)
            v_net_revenue = self.counter.calculate_v_net_revenue(self.v_net)
            return 1000 + v_net_revenue - v_net_cost
        else:
            return -float('inf')

    def random_select_next_state(self):
        """Random select a physical node to accommodate the next virtual node"""
        candidate_p_nodes = self.controller.find_candidate_nodes(
            v_net=self.v_net, 
            p_net=self.p_net, 
            v_node_id=self.v_node_id+1, 
            filter=self.selected_p_net_nodes)

        self.max_expansion = len(candidate_p_nodes)
        if self.max_expansion > 0:
            random_choice = random.choice([action for action in candidate_p_nodes])
        else:
            random_choice = -1
        next_state = copy.deepcopy(self)
        next_state.v_node_id = self.v_node_id + 1
        next_state.p_node_id = random_choice
        next_state.selected_p_net_nodes = self.selected_p_net_nodes + [random_choice]
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