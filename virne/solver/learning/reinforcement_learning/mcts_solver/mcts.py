# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import math
from threading import Thread

from virne.core import Solution
from virne.core.environment import SolutionStepEnvironment
from .node import Node, State
from virne.solver.base_solver import Solver, SolverRegistry


@SolverRegistry.register(solver_name='mcts', solver_type='r_learning')
class MctsSolver(Solver):
    """
    A Reinforcement Learning-based solver for VNE that uses Monte Carlo Tree Search (MCTS) algorithm.

    References:
        - Soroush Haeri et al. "Virtual Network Embedding via Monte Carlo Tree Search". TCYB, 2018.

    Attributes:
        computation_budget: the number of times to run the search algorithm
        exploration_constant: the exploration constant in the UCB1 formula
    """
    def __init__(self,  controller, recorder, counter, logger, config, **kwargs):
        super(MctsSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.computation_budget = kwargs.get('computation_budget', 5)
        self.exploration_constant = kwargs.get('exploration_constant', 0.5)
        # ranking strategy
        self.reusable = kwargs.get('reusable', False)
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'bfs_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        init_state = State(p_net, v_net, self.controller, self.recorder, self.counter)
        current_node = Node(None, init_state)
        solution = Solution.from_v_net(v_net)

        # node mapping
        for v_node_id in range(v_net.num_nodes):
            current_node = self.search(current_node)
            if current_node is None:
                solution['place_result'] = False
                return solution
            p_node_id = current_node.state.p_node_id
            if p_node_id == -1:
                solution['place_result'] = False
                return solution
            # place
            place_result, place_info = self.controller.node_mapper.place(v_net, p_net, v_node_id, p_node_id, solution=solution)

        # link mapping
        link_mapping_result = self.controller.link_mapper.link_mapping(v_net, p_net, solution=solution, 
                                                        shortest_method=self.shortest_method, k=self.k_shortest, inplace=True)
        if not link_mapping_result:
            solution['route_result'] = False
            return solution
        solution['result'] = True
        return solution

    def search_mp(self, node):
        # Run as much as possible under the computation budget
        search_runners = []
        for i in range(self.computation_budget):
            search_runners.append(SearchRunner(self, node))
        for search_runner in search_runners:
            search_runner.start()
        for search_runner in search_runners:
            search_runner.join()
        # N. Get the best next node
        best_next_node = self.best_child(node, False)
        return best_next_node

    def search(self, node):
        """Monte Carlo Tree Search

        Starting from a root node, find one subnode with highest exploitation value based on the exploration experience 

        1. Selection
            Select one worth node to explore. There are three types nodes: 
                (1) Never visited   (2) Uncompleted expanded  (3) Completed expanded

        2. Expansion
            Add statistic information to selected node

        3. Simulation
            Get the final reward estimating the quality of visited nodes.

        4. Backpropagation
            Update the exploitation value of visited nodes using the obtained reward

        Inference: Selected the subnode  with highest exploitation value
        """
        # Run as much as possible under the computation budget
        for i in range(self.computation_budget):
            # 1. Find the best node to expand
            expand_node = self.select_and_expand(node)
            if expand_node is None:
                break

            # 2. Random run to add node and get reward
            reward = self.simulate(expand_node)

            # 3. Update all passing nodes with reward
            self.backpropagate(expand_node, reward)

        # N. Get the best next node
        best_next_node = self.best_child(node, False)

        return best_next_node


    def select_and_expand(self, node):
        """
        Based on exploration / exploitation algorithm, get the bast node to expand
        
        (1) Preferentially, select one node never visited (if there are multiple nodes, random select one)
        (2) Otherwise, select one node with highest UCB value (exploration / exploitation)
        """

        while not node.state.is_terminal():

            if node.is_complete_expand():
                node = self.best_child(node, True)
                if node is None:
                    break
            else:
                next_node = self.expand(node)
                return next_node

        # Return the leaf node
        return node

    def expand(self, node):
        """
        Expand a node with random choice policy
        """
        tried_actions = [child_node.state.p_node_id for child_node in node.children]

        new_state = node.state.random_select_next_state()

        # 1. feasiable
        if new_state.p_node_id != -1:
            # 2. it's different from other expanded nodes
            while new_state.p_node_id in tried_actions:
                new_state = node.state.random_select_next_state()

        next_node = Node(node, new_state)
        return next_node

    def simulate(self, node):
        """
        Use the random policy to expand the node, and return the final reward
        """

        current_state = node.state

        while not current_state.is_terminal():
            # randomly select one action to play and get next state
            current_state = current_state.random_select_next_state()

        if current_state.p_node_id == -1:
            return -float('inf')
        else:
            return current_state.compute_final_reward()

    def best_child(self, node, is_exploration):
        """
        Use the UCB algorithm to blance the exploration and exploitation
        Return the node with highest value 
        (When inference phase, directly use greedy stratagy)
        """

        best_score = -float('inf')
        best_child_node = None

        # travel all child nodes to find the best one
        for child_node in node.children:

            # ignore exploration for inference
            if is_exploration:
                c = self.exploration_constant
            else:
                c = 0.0

            # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
            left = child_node.value / child_node.visit_times
            right = math.log(node.visit_times) / child_node.visit_times
            score = left + c * math.sqrt(right)

            if score > best_score:
                best_child_node = child_node
                best_score = score

        return best_child_node

    def backpropagate(self, node, reward):
        """
        Update values of the node's all predecessors using reward
        """
        # update util the root node
        while node is not None:
            # update the visit times
            node.visit_times += 1

            # update the quality value
            if reward != -float('inf'):
                node.value += reward

            # change the node to the parent node
            node = node.parent


class SearchRunner(Thread):

    def __init__(self, mcts_solver, node):
        Thread.__init__(self)
        self.mcts_solver = mcts_solver
        self.node = node

    def run(self):
        # 1. Find the best node to expand
        expand_node = self.mcts_solver.select_and_expand(self.node)
        if expand_node is None:
            return

        # 2. Random run to add node and get reward
        reward = self.mcts_solver.simulate(expand_node)

        # 3. Update all passing nodes with reward
        self.mcts_solver.backpropagate(expand_node, reward)