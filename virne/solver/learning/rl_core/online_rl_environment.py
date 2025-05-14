# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from pprint import pprint
import gym
import copy
import numpy as np
from gym import spaces
from collections import defaultdict
from virne.core import BaseEnvironment
from .rl_enviroment_base import RLBaseEnv
from ..obs_handler import ObservationHandler
from ...rank.node_rank import rank_nodes


class OnlineRLEnvBase(BaseEnvironment, RLBaseEnv):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs):
        BaseEnvironment.__init__(self, p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs)
        RLBaseEnv.__init__(self, **kwargs)

    def ready(self, event_id=0):
        super().ready(event_id)
        self.ranked_v_net_nodes = rank_nodes(self.v_net, self.config.solver.node_ranking_method)
        self.ranked_v_net_nodes = self.v_net.ranked_nodes
        self.v_net_reward = 0
        return 


class PlaceStepRLEnv(OnlineRLEnvBase):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs):
        super(PlaceStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs)

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
        # Reject Actively
        if self.allow_rejection and p_node_id == self.p_net.num_nodes:
            self.rollback_for_failure(reason='reject')
        else:
            assert p_node_id in list(self.p_net.nodes)
            # Try Deploy
            # Stage 1: Node Mapping
            node_place_result, node_place_info = self.controller.node_mapper.place(self.v_net, self.p_net, self.curr_v_node_id, p_node_id, self.solution, if_allow_constraint_violation=self.if_allow_constraint_violation)
            # Case 1: Node Place Success / Uncompleted
            if node_place_result and len(self.placed_v_net_nodes) < self.v_net.num_nodes:
                info = {**self.recorder.state, **self.solution.to_dict()}
                return self.get_observation(), self.compute_reward(info), False, self.get_info(info)
            # Case 2: Node Place Failure
            if not node_place_result:
                self.rollback_for_failure(reason='place')
            # Stage 2: Link Mapping
            # Case 3: Try Link Mapping
            if node_place_result and len(self.placed_v_net_nodes) == self.v_net.num_nodes:
                link_mapping_result = self.controller.link_mapper.link_mapping(self.v_net, 
                                                                    self.p_net, 
                                                                    solution=self.solution, 
                                                                    sorted_v_links=list(self.v_net.links), 
                                                                    shortest_method=self.shortest_method, 
                                                                    k=self.k_shortest, 
                                                                    inplace=True, 
                                                                    if_allow_constraint_violation=self.if_allow_constraint_violation)
                # Link Mapping Failure
                if not link_mapping_result:
                    self.rollback_for_failure(reason='route')
                # Success
                else:
                    self.solution['result'] = True

        record = self.recorder.count(self.v_net, self.p_net, self.solution)
        reward = self.compute_reward(record)

        extra_info = {'v_net_reward': self.v_net_reward, 'cumulative_reward': self.cumulative_reward}
        record = self.add_record(record, extra_info)
        # Leave events transition
        deploy_success = record['result']
        deploy_failure = not record['place_result'] or not record['route_result']
        if deploy_success or deploy_failure:
            done = self.transit_obs()
        else:
            done = False

        return self.get_observation(), reward, done, self.get_info(record)


class JointPRStepRLEnv(OnlineRLEnvBase):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs):
        super(JointPRStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs)

    def step(self, action):
        """
        Joint Place and Route with action p_net node.

        All possible case
            Uncompleted Success: (Node place and Link route successfully)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (Node place failed or Link route failed)
        """
        p_node_id = int(action)
        # Reject Actively
        if self.allow_rejection and p_node_id == self.p_net.num_nodes:
            self.rollback_for_failure(reason='reject')
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
                                                                            if_allow_constraint_violation=self.if_allow_constraint_violation)
            # Step Failure
            if not place_and_route_result:
                failure_reason = self.get_failure_reason(self.solution)
                self.rollback_for_failure(failure_reason)
            else:
                # VN Success ?
                if self.num_placed_v_net_nodes == self.v_net.num_nodes:
                    self.solution['result'] = True
                # Step Success
                else:
                    solution_info = self.recorder.counter.count_partial_solution(self.v_net, self.solution)
                    info = {**self.recorder.state, **solution_info}
                    return self.get_observation(), self.compute_reward(info), False, self.get_info(info)

        record = self.recorder.count(self.v_net, self.p_net, self.solution)
        reward = self.compute_reward(record)
        extra_info = {'v_net_reward': self.v_net_reward, 'cumulative_reward': self.cumulative_reward}
        record = self.add_record(record, extra_info)

        # Leave events transition
        if self.solution['early_rejection'] or not place_and_route_result or self.solution['result']:
            done = self.transit_obs()
        else:
            done = False

        return self.get_observation(), reward, done, self.get_info(record)


class NodePairStepRLEnv(JointPRStepRLEnv):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs):
        super(NodePairStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs)

    def ready(self, event_id=0):
        super().ready(event_id)
        self.curr_v_node_id = None
    
    @property
    def curr_v_node_id(self):
        return self._curr_v_node_id

    def generate_action_mask(self):
        mask = np.zeros([self.v_net.num_nodes, self.p_net.num_nodes])
        for v_node_id, p_candidates in self.candidates_dict.items():
            mask[v_node_id][p_candidates] = 1
        # Each virtual node only can be changed once
        for v_node_id, p_id in self.solution['node_slots'].items():
            mask[:, p_id] = 0
        if mask.sum() == 0:
            mask[0][0] = 1
        # mask = mask.reshape(mask.shape[1], mask.shape[0])
        return mask.flatten()

    def step(self, action):
        """
        Joint Place and Route with action p_net node.

        All possible case
            Uncompleted Success: (Node place and Link route successfully)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (Node place failed or Link route failed)
        """
        self._curr_v_node_id = v_node_id
        v_node_id = int(action)
        p_node_id = int(action[1])
        return super().step(p_node_id)

class SolutionStepRLEnv(OnlineRLEnvBase):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs):
        super(SolutionStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs)

    def step(self, action):
        solution = action
        # Success
        if solution['result']:
            self.solution = solution
            self.solution['info'] = 'Success'
        # Failure
        else:
            failure_reason = self.get_failure_reason(solution)
            self.rollback_for_failure(reason=failure_reason)

        record = self.recorder.count(self.v_net, self.p_net, self.solution)
        reward = self.compute_reward(record)
        extra_info = {'v_net_reward': self.v_net_reward, 'cumulative_reward': self.cumulative_reward}
        record = self.add_record(record, extra_info)

        done = self.transit_obs()
        return self.get_observation(), reward, done, self.get_info(record)

    def action_masks(self):
        return self.generate_action_mask()

    def generate_action_mask(self):
        return np.ones(2, dtype=bool)


class NodeSlotsStepRLEnv(OnlineRLEnvBase):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs):
        super(NodeSlotsStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs)

    def step(self, action):
        node_slots = action

        if len(node_slots) == self.v_net.num_nodes:
            self.solution['node_slots'] = node_slots
            link_mapping_result = self.controller.link_mapper.link_mapping(self.v_net, 
                                                                self.p_net, 
                                                                solution=self.solution, 
                                                                sorted_v_links=list(self.v_net.links), 
                                                                shortest_method=self.shortest_method, 
                                                                k=self.k_shortest, 
                                                                inplace=True,
                                                                if_allow_constraint_violation=self.if_allow_constraint_violation)
            # Link Mapping Failure
            if not link_mapping_result:
                self.rollback_for_failure(reason='route')
            # Success
            else:
                self.solution['result'] = True
        else:
            self.rollback_for_failure(reason='place')

        record = self.recorder.count(self.v_net, self.p_net, self.solution)
        reward = self.compute_reward(record)
        extra_info = {'v_net_reward': self.v_net_reward, 'cumulative_reward': self.cumulative_reward}
        record = self.add_record(record, extra_info)

        done = self.transit_obs()
        return self.get_observation(), reward, done, self.get_info(record)


if __name__ == '__main__':
    pass
    