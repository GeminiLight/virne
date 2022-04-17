import numpy as np
import networkx as nx

from ..solver import Solver


class FitSolver(Solver):

    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        super().__init__(name, reusable, verbose, **kwargs)


class FirstFitSolver(FitSolver):

    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(FirstFitSolver, self).__init__(name='first_fit', reusable=reusable, verbose=verbose, **kwargs)

    def solve(self, instance):
        vn, pn  = instance['vn'], instance['pn']

    def select_action(self, env, vnf_id=None):
        if vnf_id is None:
            vnf_id = env.curr_vnf_id
        vnf_requests_dict = env.curr_vn.nodes[vnf_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(vnf_requests_dict, filter=filter, rtype='id')
        if len(candidate_nodes) == 0:
            return 0
        else:
            return candidate_nodes[0]


class RandomFitSolver(Solver):

    def __init__(self, reusable=False, **kwargs):
        super(RandomFitSolver, self).__init__(name='random_fit', type='heuristic', mapping='joint', reusable=reusable)

    def select_action(self, env, vnf_id=None):
        if vnf_id is None:
            vnf_id = env.curr_vnf_id
        vnf_requests_dict = env.curr_vn.nodes[vnf_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(vnf_requests_dict, filter=filter, rtype='id')
        if len(candidate_nodes) == 0:
            return 0
        else:
            random_id = np.random.choice(len(candidate_nodes), 1)[0]
            p_node_id = candidate_nodes[random_id]
            return p_node_id


class NearestFitSolver(Solver):
    
    def __init__(self, reusable=False, **kwargs):
        super(NearestFitSolver, self).__init__(name='nearest_fit', type='heuristic', mapping='joint', reusable=reusable)

    def select_action(self, env, vnf_id=None):
        if vnf_id is None:
            vnf_id = env.curr_vnf_id
        vnf_requests_dict = env.curr_vn.nodes[vnf_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(vnf_requests_dict, filter=filter, rtype='id')
        
        # first vnf
        if vnf_id == 0:
            if len(candidate_nodes) == 0:
                # FAILURE
                return env.pn.num_nodes
            random_id = np.random.choice(len(candidate_nodes), 1)[0]
            p_node_id = candidate_nodes[random_id]
            return p_node_id

        # !!!
        temp_graph = self.pn
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
            return env.pn.num_nodes
        except:
            # FAILURE
            return env.pn.num_nodes


if __name__ == '__main__':
    pass
