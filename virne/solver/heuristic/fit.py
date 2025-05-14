# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import numpy as np
import networkx as nx

from virne.solver.base_solver import Solver, SolverRegistry


class FitSolver(Solver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)


class FirstFitSolver(FitSolver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(FirstFitSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)

    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']

    def select_action(self, env, v_node_id=None):
        if v_node_id is None:
            v_node_id = env.curr_v_node_id
        v_node_requests_dict = env.v_net.nodes[v_node_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(v_node_requests_dict, filter=filter, rtype='id')
        if len(candidate_nodes) == 0:
            return 0
        else:
            return candidate_nodes[0]


class RandomFitSolver(Solver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(RandomFitSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)

    def select_action(self, env, v_node_id=None):
        if v_node_id is None:
            v_node_id = env.curr_v_node_id
        v_node_requests_dict = env.v_net.nodes[v_node_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(v_node_requests_dict, filter=filter, rtype='id')
        if len(candidate_nodes) == 0:
            return 0
        else:
            random_id = np.random.choice(len(candidate_nodes), 1)[0]
            p_node_id = candidate_nodes[random_id]
            return p_node_id


class NearestFitSolver(Solver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(NearestFitSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)

    def select_action(self, env, v_node_id=None):
        if v_node_id is None:
            v_node_id = env.curr_v_node_id
        v_node_requests_dict = env.v_net.nodes[v_node_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(v_node_requests_dict, filter=filter, rtype='id')
        
        # first v_node
        if v_node_id == 0:
            if len(candidate_nodes) == 0:
                # FAILURE
                return env.p_net.num_nodes
            random_id = np.random.choice(len(candidate_nodes), 1)[0]
            p_node_id = candidate_nodes[random_id]
            return p_node_id

        # !!!
        temp_graph = self.p_net
        # look for the shortest path
        last_pid = env.selected_nodes[-1]
        try:
            path_length = nx.shortest_path_length(temp_graph, source=last_pid)
            sorted_nodes = [node for node, _ in sorted(path_length.items(), key=lambda item: item[1])]
            for node in sorted_nodes:
                if node in candidate_nodes:
                    return node
                else:
                    continue
            # FAILURE
            return env.p_net.num_nodes
        except:
            # FAILURE
            return env.p_net.num_nodes


if __name__ == '__main__':
    pass
