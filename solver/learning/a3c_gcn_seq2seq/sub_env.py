# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import numpy as np
import networkx as nx
from gym import spaces
from ..sub_rl_environment import JointPRStepSubRLEnv, PlaceStepSubRLEnv


class SubEnv(JointPRStepSubRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(SubEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        num_p_net_node_attrs = len(self.p_net.get_node_attrs(['resource', 'extrema']))
        num_p_net_link_attrs = len(self.p_net.get_link_attrs(['resource', 'extrema']))
        num_p_net_features = num_p_net_node_attrs + 1
        # self.observation_space = spaces.Dict({
        #     'p_net_x': spaces.Box(low=0, high=1, shape=(self.p_net.num_nodes, num_p_net_features), dtype=np.float32),
        #     'p_net_edge_index': spaces.Box(low=0, high=self.p_net.num_nodes, shape=(2, self.p_net.num_links), dtype=np.int32),
        #     'p_net_edge_attr': spaces.Box(low=0, high=self.p_net.num_nodes, shape=(self.p_net.num_links, 2), dtype=np.int32),
        #     'v_node': spaces.Box(low=0, high=100, shape=(int(num_p_net_node_attrs/2) + int(num_p_net_link_attrs/2) + 2, ), dtype=np.float32)
        # })
        
    def compute_reward(self, solution):
        """Calculate deserved reward according to the result of taking action."""
        if solution['result'] :
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = 1 / self.v_net.num_nodes
        else:
            reward = - 1 / self.v_net.num_nodes
        self.solution['v_net_reward'] += reward
        return reward

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        # sub - attr
        return {'p_net_x': p_net_obs['x'],
                'p_net_edge_index': p_net_obs['edge_index'],
                'v_net_x': v_net_obs['x']}

    def _get_p_net_obs(self):
        """
        node_resource, average_distance, p_net_degreees, p_net_nodes_states, v_node_features
        """
        # node data
        node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=['resource', 'extrema'], node_attr_benchmarks=self.node_attr_benchmarks)
        edge_aggr_data = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource', 'extrema'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        selected_nodes = np.zeros(self.p_net.num_nodes, dtype=np.float32)
        selected_nodes[self.selected_p_net_nodes] = 1.
        node_data = np.concatenate((node_data, edge_aggr_data, np.expand_dims(selected_nodes, axis=-1)), axis=-1)
        # edge_index
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        # data
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
        }
        return p_net_obs

    def _get_v_net_obs(self):
        if self.curr_v_node_id  >= self.v_net.num_nodes:
            return []
        node_data = self.obs_handler.get_node_attrs_obs(self.v_net, node_attr_types=['resource'], node_attr_benchmarks=self.node_attr_benchmarks)
        edge_aggr_data = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        node_data = np.concatenate((node_data, edge_aggr_data), axis=-1)
        return {'x': node_data}