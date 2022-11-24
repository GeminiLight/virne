from pprint import pprint
import gym
import copy
import numpy as np
from gym import spaces
from collections import defaultdict
from base import Environment
from .obs_handler import ObservationHandler
from ..rank.node_rank import rank_nodes


class VNERLEnv(gym.Env):

    def __init__(self, allow_rejection=False, allow_revocable=False, **kwargs):
        super(VNERLEnv, self).__init__()
        self.allow_rejection = allow_rejection
        self.allow_revocable = allow_revocable
        self.rejection_action = self.p_net.num_nodes - 1 + int(self.allow_rejection) if allow_rejection else None
        self.revocable_action = self.p_net.num_nodes - 1 + int(self.allow_revocable) + int(self.allow_rejection) if allow_revocable else None
        self.num_actions = self.p_net.num_nodes + int(allow_rejection) + int(allow_revocable)
        self.action_space = spaces.Discrete(self.num_actions)
        self.obs_handler = ObservationHandler()
        _, self.node_attrs_benchmark = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=['extrema'], normalization=True)
        _, self.link_attrs_benchmark = self.obs_handler.get_link_attrs_obs(self.p_net, link_attr_types=['extrema'], normalization=True)
        _, self.link_aggr_attrs_benchmark = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['extrema'], normalization=True)
        self.reward_weight = kwargs.get('reward_weight', 0.1)

        # for revocable action
        self.revoked_actions_dict = defaultdict(list)

    def if_rejection(self, action):
        return self.allow_rejection and action == self.rejection_action

    def if_revocable(self, action):
        return self.revocable_action and action == self.revocable_action

    def step(self, action):
       return NotImplementedError

    def compute_reward(self,):
        return NotImplementedError

    def get_observation(self):
        return NotImplementedError

    def get_info(self, record={}):
        info = copy.deepcopy(record)
        return info

    def get_curr_place_progress(self):
        return self.num_placed_v_net_nodes / (self.v_net.num_nodes - 1)

    def get_node_load_balance(self, p_node_id):
        n_attrs = self.p_net.get_node_attrs(['resource'])
        if len(n_attrs) > 1:
            n_resources = np.array([self.p_net.nodes[p_node_id][n_attr.name] for n_attr in n_attrs])
            load_balance = np.std(n_resources)
        else:
            n_attr = self.p_net.get_node_attrs(['extrema'])[0]
            load_balance = self.p_net.nodes[p_node_id][n_attr.originator] / self.p_net.nodes[p_node_id][n_attr.name]
        return load_balance

    def generate_action_mask(self):
        candidate_nodes = self.controller.find_candidate_nodes(self.v_net, self.p_net, self.curr_v_node_id, filter=self.selected_p_net_nodes)
        # candidate_nodes = self.controller.find_feasible_nodes(self.p_net, self.v_net, self.curr_v_node_id, self.solution['node_slots'])
        mask = np.zeros(self.num_actions, dtype=bool)
        # add special actions
        if self.allow_rejection:
            candidate_nodes.append(self.rejection_action)
        if self.allow_revocable:
            if self.num_placed_v_net_nodes != 0:
                candidate_nodes.append(self.revocable_action)
            revoked_actions = self.revoked_actions_dict[(str(self.solution.node_slots), self.curr_v_node_id)]
            [candidate_nodes.remove(a_id) for a_id in revoked_actions if a_id in revoked_actions]
        mask[candidate_nodes] = True
        # if mask.sum() == 0: 
            # mask[0] = True
        return mask

    def action_masks(self):
        return self.generate_action_mask()

    @property
    def selected_p_net_nodes(self):
        return list(self.solution['node_slots'].values())

    @property
    def placed_v_net_nodes(self):
        return list(self.solution['node_slots'].keys())

    @property
    def num_placed_v_net_nodes(self):
        return len(self.solution['node_slots'].keys())

    @property
    def last_placed_v_node_id(self):
        if self.num_placed_v_net_nodes == 0:
            return None
        return list(self.solution['node_slots'].keys())[-1]

    @property
    def curr_v_node_id(self):
        if self.num_placed_v_net_nodes == self.v_net.num_nodes:
            return 0
        return self.v_net.ranked_nodes[self.num_placed_v_net_nodes]


class BasicRLEnv(Environment, VNERLEnv):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        Environment.__init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs)
        VNERLEnv.__init__(self, **kwargs)

    def ready(self, event_id=0):
        self.ranked_v_net_nodes = rank_nodes(self.v_net, self.node_ranking_method)
        self.ranked_v_net_nodes = self.v_net.ranked_nodes
        self.v_net_reward = 0
        return super().ready(event_id)


class PlaceStepRLEnv(BasicRLEnv):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(PlaceStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)

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
            node_place_result, node_place_info = self.controller.place(self.v_net, self.p_net, self.curr_v_node_id, p_node_id, self.solution)
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
                link_mapping_result = self.controller.link_mapping(self.v_net, 
                                                                    self.p_net, 
                                                                    solution=self.solution, 
                                                                    sorted_v_links=list(self.v_net.links), 
                                                                    shortest_method=self.shortest_method, 
                                                                    k=self.k_shortest, 
                                                                    inplace=True)
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
            if done:
                summary_info = self.summary_records()
        else:
            done = False

        return self.get_observation(), reward, done, self.get_info(record)


class JointPRStepRLEnv(BasicRLEnv):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(JointPRStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)

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
                                                                            k=self.k_shortest)
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
            if done:
                summary_info = self.summary_records()
        else:
            done = False

        return self.get_observation(), reward, done, self.get_info(record)


class PairStepRLEnv(BasicRLEnv):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(PairStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)

    def step(self, action):
        """
        Joint Place and Route with action p_net node.

        All possible case
            Uncompleted Success: (Node place and Link route successfully)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (Node place failed or Link route failed)
        """
        v_node_id = int(action[0])
        p_node_id = int(action[1])
        # Reject Actively
        if self.allow_rejection and p_node_id == self.p_net.num_nodes:
            self.rollback_for_failure(reason='reject')
        else:
            assert p_node_id in list(self.p_net.nodes)
            place_and_route_result, place_and_route_info = self.controller.place_and_route(
                                                                                self.v_net, 
                                                                                self.p_net, 
                                                                                v_node_id, 
                                                                                p_node_id, 
                                                                                self.solution, 
                                                                                shortest_method=self.shortest_method, 
                                                                                k=self.k_shortest)
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
        if not place_and_route_result or self.solution['early_reject'] or self.solution['result']:
            done = self.transit_obs()
            if done:
                summary_info = self.summary_records()
        else:
            done = False

        return self.get_observation(), reward, done, self.get_info(record)


class SolutionStepRLEnv(BasicRLEnv):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(SolutionStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)

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
        if done:
            summary_info = self.summary_records()
        return self.get_observation(), reward, done, self.get_info(record)

    def action_masks(self):
        return self.generate_action_mask()

    def generate_action_mask(self):
        return np.ones(2, dtype=bool)


class NodeSlotsStepRLEnv(BasicRLEnv):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(NodeSlotsStepRLEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)

    def step(self, action):
        node_slots = action

        if len(node_slots) == self.v_net.num_nodes:
            self.solution['node_slots'] = node_slots
            link_mapping_result = self.controller.link_mapping(self.v_net, 
                                                                self.p_net, 
                                                                solution=self.solution, 
                                                                sorted_v_links=list(self.v_net.links), 
                                                                shortest_method=self.shortest_method, 
                                                                k=self.k_shortest, 
                                                                inplace=True)
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
        if done:
                summary_info = self.summary_records()
        return self.get_observation(), reward, done, self.get_info(record)


if __name__ == '__main__':
    pass
    