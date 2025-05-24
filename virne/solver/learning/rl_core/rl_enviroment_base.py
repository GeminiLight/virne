# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from pprint import pprint
import gym
import copy
import numpy as np
import networkx as nx
from gym import spaces
from collections import defaultdict

from virne.network.attribute.attribute_benchmark_manager import AttributeBenchmarkManager
from ..obs_handler import ObservationHandler
from ...rank.node_rank import rank_nodes
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Solution


class RLBaseEnv(gym.Env):

    p_net: PhysicalNetwork
    v_net: VirtualNetwork
    controller: Controller
    recorder: Recorder
    counter: Counter
    solution: Solution

    def __init__(self, allow_rejection=False, allow_revocable=False, **kwargs):
        super(RLBaseEnv, self).__init__()
        self.obs_handler = ObservationHandler()
        self.allow_rejection = allow_rejection
        self.allow_revocable = allow_revocable
        self.rejection_action = self.p_net.num_nodes - 1 + int(self.allow_rejection) if allow_rejection else None
        self.revocable_action = self.p_net.num_nodes - 1 + int(self.allow_revocable) + int(self.allow_rejection) if allow_revocable else None
        self.num_actions = self.p_net.num_nodes + int(allow_rejection) + int(allow_revocable)
        self.action_space = spaces.Discrete(self.num_actions)
        # for revocable action
        self.if_allow_constraint_violation = kwargs.get('if_allow_constraint_violation', False)
        self.revoked_actions_dict = defaultdict(list)
        self.extra_info_dict = {}

    def reset(self):
        self.extra_info_dict = {}
        self.revoked_actions_dict = defaultdict(list)
        return self.get_observation()

    def if_rejection(self, action):
        return self.allow_rejection and action == self.rejection_action

    def if_revocable(self, action):
        return self.revocable_action and action == self.revocable_action

    def step(self, action):
       raise NotImplementedError

    def compute_reward(self,):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_info(self, record={}):
        info = copy.deepcopy(record)
        for k, v in self.extra_info_dict.items():
            info[k] = v
        return info

    def add_extra_info(self, info_dict):
        self.extra_info_dict.update(info_dict)

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
        node_slots = self.solution['node_slots']
        return len(node_slots.keys())

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