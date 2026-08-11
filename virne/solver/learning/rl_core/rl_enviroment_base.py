# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from pprint import pprint
import copy
import numpy as np
import networkx as nx
from gymnasium import Env, spaces
from collections import defaultdict

from virne.network.attribute.attribute_benchmark_manager import AttributeBenchmarkManager
from ..obs_handler import ObservationHandler
from ...rank.node_rank import rank_nodes
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Solution


class RLBaseEnv(Env):

    p_net: PhysicalNetwork
    v_net: VirtualNetwork
    controller: Controller
    recorder: Recorder
    counter: Counter
    solution: Solution

    def __init__(self, allow_rejection=False, allow_revocable=False, **kwargs):
        super().__init__()
        self.obs_handler = ObservationHandler()
        self._refresh_node_indices()
        self.allow_rejection = allow_rejection
        self.allow_revocable = allow_revocable
        self.rejection_action = self.p_net.num_nodes if allow_rejection else None
        self.revocable_action = self.p_net.num_nodes + int(allow_rejection) if allow_revocable else None
        self.num_actions = self.p_net.num_nodes + int(allow_rejection) + int(allow_revocable)
        self.action_space = spaces.Discrete(self.num_actions)
        # for revocable action
        self.if_allow_constraint_violation = kwargs.get('if_allow_constraint_violation', False)
        self.revoked_actions_dict = defaultdict(list)
        self.extra_info_dict = {}
        self._no_feasible_action = False

    def _refresh_node_indices(self):
        self.p_node_ids = list(self.p_net.nodes)
        self.p_node_id_to_action = {
            node_id: action for action, node_id in enumerate(self.p_node_ids)
        }
        self.v_node_ids = list(self.v_net.nodes) if hasattr(self, 'v_net') else []
        self.v_node_id_to_index = {
            node_id: index for index, node_id in enumerate(self.v_node_ids)
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.extra_info_dict = {}
        self.revoked_actions_dict = defaultdict(list)
        return self.get_observation(), {}

    def _make_step_result(self, reward, terminated, info, *, truncated=False):
        """Build a Gymnasium step result from the environment's current state."""
        return (
            self.get_observation(),
            reward,
            terminated,
            truncated,
            self.get_info(info),
        )

    def if_rejection(self, action):
        return self.allow_rejection and action == self.rejection_action

    def if_revocable(self, action):
        return bool(self.allow_revocable and action == self.revocable_action)

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
        candidate_nodes = list(self.controller.find_candidate_nodes(
            self.v_net,
            self.p_net,
            self.curr_v_node_id,
            filter=self.selected_p_net_nodes,
        ))
        # candidate_nodes = self.controller.find_feasible_nodes(self.p_net, self.v_net, self.curr_v_node_id, self.solution['node_slots'])
        mask = np.zeros(self.num_actions, dtype=bool)
        self._no_feasible_action = False
        if self.allow_revocable:
            revoked_actions = self.revoked_actions_dict[
                (str(self.solution.node_slots), self.curr_v_node_id)
            ]
            candidate_nodes = [
                node_id for node_id in candidate_nodes
                if node_id not in revoked_actions
            ]

        try:
            candidate_actions = [
                self.p_node_id_to_action[node_id] for node_id in candidate_nodes
            ]
        except KeyError as exc:
            raise ValueError(
                f'Controller returned unknown physical node ID: {exc.args[0]!r}'
            ) from exc

        # add special actions
        if self.allow_rejection:
            candidate_actions.append(self.rejection_action)
        if self.allow_revocable and self.num_placed_v_net_nodes != 0:
            candidate_actions.append(self.revocable_action)
        if candidate_actions:
            mask[candidate_actions] = True
        else:
            # Categorical policies cannot represent an empty action set. Expose a
            # deterministic sentinel action and terminate it in the environment
            # before invoking the controller.
            mask[0] = True
            self._no_feasible_action = True
        return mask

    def action_to_p_node_id(self, action):
        action = int(action)
        if action < 0 or action >= self.p_net.num_nodes:
            raise ValueError(f'Action {action} is not a physical-node action')
        return self.p_node_ids[action]

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
            return self.v_node_ids[0]
        if hasattr(self.v_net, 'node_ranking'):
            ranked_nodes = list(self.v_net.node_ranking)
        else:
            ranked_nodes = list(self.v_net.ranked_nodes)
        return ranked_nodes[self.num_placed_v_net_nodes]

    @property
    def curr_v_node_index(self):
        return self.v_node_id_to_index[self.curr_v_node_id]
