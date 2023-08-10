# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import numpy as np
import networkx as nx

from virne.base import Solution
from .rl_enviroment_base import RLBaseEnv
from .online_rl_environment import *
from ..obs_handler import ObservationHandler
from ...rank.node_rank import rank_nodes
from virne.config import set_sim_info_to_object


class InstanceRLEnv(RLBaseEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        self.p_net = p_net
        self.v_net = v_net
        self.counter = counter
        self.recorder = recorder
        self.controller = controller
        super(InstanceRLEnv, self).__init__(**kwargs)
        self.p_net_backup = copy.deepcopy(self.p_net)
        self.solution = Solution(v_net)
        # ranking strategy
        self.reusable = kwargs.get('reusable', False)
        self.node_ranking_method = kwargs.get('node_ranking_method', 'order')
        self.link_ranking_method = kwargs.get('link_ranking_method', 'order')
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'bfs_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)
        # calcuate graph metrics
        # self.node_ranking_method = 'nps'
        self.ranked_v_net_nodes = rank_nodes(self.v_net, self.node_ranking_method)
        self.solution['v_net_reward'] = 0
        self.solution['num_interactions'] = 0
        # set_sim_info_to_object(kwargs, self)

    def reset(self):
        self.solution.reset()
        self.p_net = copy.deepcopy(self.p_net_backup)
        return super().reset()

    def reject(self):
        self.solution['early_rejection'] = True
        solution_info = self.solution.to_dict()
        done = True
        return self.get_observation(), self.compute_reward(self.solution), done, self.get_info(solution_info)

    def revoke(self):
        assert len(self.placed_v_net_nodes) != 0
        self.solution['place_result'] = True
        self.solution['route_result'] = True
        self.solution['revoke_times'] += 1
        last_v_node_id = self.placed_v_net_nodes[-1]
        paired_p_node_id = self.selected_p_net_nodes[-1]
        self.controller.undo_place_and_route(self.v_net, self.p_net, last_v_node_id, paired_p_node_id, self.solution)
        solution_info = self.counter.count_partial_solution(self.v_net, self.solution)
        self.revoked_actions_dict[str(self.solution.node_slots), last_v_node_id].append(paired_p_node_id)
        return self.get_observation(), self.compute_reward(self.solution), False, self.get_info(solution_info)


class SolutionStepInstanceRLEnv(InstanceRLEnv):
    
    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(SolutionStepInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)

    def step(self, solution):
        # Success
        if solution['result']:
            self.solution = solution
            self.solution['description'] = 'Success'
        # Failure
        else:
            solution = Solution(self.v_net)
        return self.get_observation(), self.compute_reward(), True, self.get_info(solution.to_dict())

    def get_observation(self):
        return {'v_net': self.v_net, 'p_net': self.p_net}

    def compute_reward(self):
        return 0



class PlaceStepInstanceRLEnv(InstanceRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(PlaceStepInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)

    def step(self, action):
        """
        Two stage: Node Mapping and Link Mapping

        All possible case
            Early Rejection: (rejection_action)
            Uncompleted Success: (Node place)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (not Node place, not Link mapping)
        """
        p_node_id = int(action)
        done = True
        # Case: Reject
        if self.if_rejection(action):
            return self.reject()
        # Case: Revoke
        if self.if_revocable(action):
            return self.revoke()
        # Case: Place in one same node
        elif not self.reusable and p_node_id in self.selected_p_net_nodes:
            self.solution['place_result'] = False
            solution_info = self.solution.to_dict()
        # Case: Try to Place
        else:
            assert p_node_id in list(self.p_net.nodes)
            # Stage 1: Node Mapping
            node_place_result, node_place_info = self.controller.place(self.v_net, self.p_net, self.curr_v_node_id, p_node_id, self.solution)
            # Case 1: Node Place Success / Uncompleted
            if node_place_result and self.num_placed_v_net_nodes < self.v_net.num_nodes:
                done = False
                solution_info = self.solution.to_dict()
                return self.get_observation(), self.compute_reward(self.solution), False, self.get_info(self.solution.to_dict())
            # Case 2: Node Place Failure
            if not node_place_result:
                self.solution['place_result'] = False
                solution_info = self.counter.count_solution(self.v_net, self.solution)
                # solution_info = self.solution.to_dict()
            # Stage 2: Link Mapping
            # Case 3: Try Link Mapping
            if node_place_result and self.num_placed_v_net_nodes == self.v_net.num_nodes:
                link_mapping_result = self.controller.link_mapping(self.v_net, 
                                                                    self.p_net, 
                                                                    solution=self.solution, 
                                                                    sorted_v_links=list(self.v_net.links), 
                                                                    shortest_method=self.shortest_method, 
                                                                    k=self.k_shortest, 
                                                                    inplace=True,
                                                                    check_feasibility=self.check_feasibility)
                # Link Mapping Failure
                if not link_mapping_result:
                    self.solution['route_result'] = False
                    solution_info = self.counter.count_solution(self.v_net, self.solution)
                    # solution_info = self.solution.to_dict()
                # Success
                else:
                    self.solution['result'] = True
                    solution_info = self.counter.count_solution(self.v_net, self.solution)
        if done:
            pass
        return self.get_observation(), self.compute_reward(solution_info), done, self.get_info(solution_info)


class JointPRStepInstanceRLEnv(InstanceRLEnv):
    
    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(JointPRStepInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)

    def step(self, action):
        """
        Joint Place and Route with action p_net node.

        All possible case
            Uncompleted Success: (Node place and Link route successfully)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (Node place failed or Link route failed)
        """
        self.solution['num_interactions'] += 1
        p_node_id = int(action)
        self.solution.selected_actions.append(p_node_id)
        if self.solution['num_interactions'] > 10 * self.v_net.num_nodes:
            # self.solution['description'] += 'Too Many Revokable Actions'
            return self.reject()
        # Case: Reject
        if self.if_rejection(action):
            return self.reject()
        # Case: Revoke
        if self.if_revocable(action):
            return self.revoke()
        # Case: reusable = False and place in one same node
        elif not self.reusable and (p_node_id in self.selected_p_net_nodes):
            self.solution['place_result'] = False
            solution_info = self.counter.count_solution(self.v_net, self.solution)
            done = True
            # solution_info = self.solution.to_dict()
        # Case: Try to Place and Route
        else:
            assert p_node_id in list(self.p_net.nodes)
            place_and_route_result, place_and_route_info = self.controller.place_and_route(
                                                                                self.v_net, 
                                                                                self.p_net, 
                                                                                self.curr_v_node_id, 
                                                                                p_node_id,
                                                                                self.solution, 
                                                                                shortest_method=self.shortest_method, 
                                                                                k=self.k_shortest,
                                                                                check_feasibility=self.check_feasibility)
            # Step Failure
            if not place_and_route_result:
                if self.allow_revocable and self.solution['num_interactions'] <= self.v_net.num_nodes * 10:
                    self.solution['selected_actions'].append(self.revocable_action)
                    return self.revoke()
                else:
                    solution_info = self.counter.count_solution(self.v_net, self.solution)
                    done = True
    
                # solution_info = self.solution.to_dict()
            else:
                # VN Success ?
                if self.num_placed_v_net_nodes == self.v_net.num_nodes:
                    self.solution['result'] = True
                    solution_info = self.counter.count_solution(self.v_net, self.solution)
                    done = True
                # Step Success
                else:
                    done = False
                    solution_info = self.counter.count_partial_solution(self.v_net, self.solution)
                    
        if done:
            pass
        
        # print(f'{t2-t1:.6f}={t3-t1:.6f}+{t2-t3:.6f}')
        return self.get_observation(), self.compute_reward(self.solution), done, self.get_info(solution_info)

    def compute_reward(self, solution):
        """Calculate deserved reward according to the result of taking action."""
        reward_weight = 0.1
        if solution['result'] :
            # node_load_balance = self.get_node_load_balance(self.selected_p_net_nodes[-1])
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            # curr_place_progress = self.get_curr_place_progress()
            # node_load_balance = self.get_node_load_balance(self.selected_p_net_nodes[-1])
            # reward = curr_place_progress * (solution['v_net_r2c_ratio']) #  - 0.01 * node_load_balance
            # reward = curr_place_progress * ((solution['v_net_r2c_ratio']) + 0.01 * node_load_balance)
            # weight = (1 + len(self.v_net.adj[curr_v_node_id - 1])) / (self.v_net.num_nodes + self.v_net.num_links)
            # reward = 0.01
            # weight = (1 + len(self.v_net.adj[curr_v_node_id - 1])) / 10
            # reward = weight * solution['v_net_r2c_ratio']
            # + 0.01 * node_load_balance
            reward = 0.
        else:
            curr_place_progress = self.get_curr_place_progress()
            # reward = - self.get_curr_place_progress()
            # weight = (1 + len(self.v_net.adj[curr_v_node_id - 1])) / (self.v_net.num_nodes + self.v_net.num_links)
            # weight = (1 + len(self.v_net.adj[curr_v_node_id])) / 10
            reward = - curr_place_progress
            # reward = - 0.1
            # reward = -1. * weight
        # reward = solution['v_net_r2c_ratio'] if solution['result'] else 0
        # reward = self.v_net.num_nodes / 10 * reward
        # reward = self.v_net.total_resource_demand / 500 * reward
        self.solution['v_net_reward'] += reward
        return reward


class NodePairStepInstanceRLEnv(JointPRStepInstanceRLEnv):
    
    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(JointPRStepInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        self._curr_v_node_id = 0
        self.candidates_dict = self.controller.construct_candidates_dict(self.v_net, self.p_net)

    @property
    def curr_v_node_id(self):
        return self._curr_v_node_id

    def generate_action_mask(self):
        mask = np.zeros([self.v_net.num_nodes, self.p_net.num_nodes])
        for v_node_id, p_candicates in self.candidates_dict.items():
            mask[v_node_id][p_candicates] = 1
        # Each virtual node only can be changed once
        for v_node_id, p_id in self.solution['node_slots'].items():
            mask[:, p_id] = 0
            mask[v_node_id, :] = 0
        if mask.sum() == 0:
            mask[0][0] = 1
        mask = mask.T
        return mask.flatten()

    def step(self, action):
        """
        Joint Place and Route with action p_net node.

        All possible case
            Uncompleted Success: (Node place and Link route successfully)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (Node place failed or Link route failed)
        """
        # The action is sampled from a heatmap with the shape of [p_net.num_nodes, v_net.num_nodes]
        p_node_id = action // self.v_net.num_nodes
        v_node_id = action % self.v_net.num_nodes

        self._curr_v_node_id = v_node_id
        # print(f'action ({action}) - v_node_id: {v_node_id}, p_node_id: {p_node_id}')
        if v_node_id in self.solution['node_slots']:
            print(f'v_node_id {v_node_id} has been placed') if v_node_id != 0 else None
            self.solution['place_result'] = False
            solution_info = self.counter.count_solution(self.v_net, self.solution)
            done = True
            return self.get_observation(), self.compute_reward(self.solution), done, self.get_info(solution_info)
        return super().step(p_node_id)


class NodeSlotsStepInstanceRLEnv(InstanceRLEnv):
    
    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(NodeSlotsStepInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)

    def step(self, node_slots):
        if len(node_slots) == self.v_net.num_nodes:
            self.controller.deploy_with_node_slots(
                self.v_net, self.p_net, 
                node_slots, self.solution, 
                inplace=True, 
                shortest_method=self.shortest_method, 
                k_shortest=self.k_shortest,
                check_feasibility=self.check_feasibility
            )
            self.counter.count_solution(self.v_net, self.solution)
            # Success
            if self.solution['result']:
                self.solution['description'] = 'Success'
        return self.get_observation(), self.compute_reward(self.solution), True, self.get_info(self.solution.to_dict())
