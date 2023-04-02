# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import numpy as np
import networkx as nx
from gym import spaces
from ..sub_rl_environment import JointPRStepSubRLEnv, PlaceStepSubRLEnv


class SubEnv(PlaceStepSubRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(SubEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        num_p_net_node_attrs = len(self.p_net.get_node_attrs(['resource']))
        num_p_net_link_attrs = len(self.p_net.get_link_attrs(['resource']))
        num_obs_attrs = num_p_net_node_attrs + num_p_net_link_attrs + 2
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.p_net.num_nodes, num_obs_attrs, 1), dtype=np.float32)
        p_net_node_degrees = np.array(list(nx.degree_centrality(self.p_net).values()))
        self.p_net_node_degrees = (p_net_node_degrees - p_net_node_degrees.min()) / (p_net_node_degrees.max() - p_net_node_degrees.min())

    def compute_reward(self, solution):
        """Calculate deserved reward according to the result of taking action."""
        if solution['result'] :
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = 0
        else:
            reward = 0
        return reward

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        return {'obs': p_net_obs}
    
    def _get_p_net_obs(self):
        # (cpu_remain, degree, sum_bw, avg_dst)
        # resource
        n_attrs = self.p_net.get_node_attrs(['resource'])
        node_attrs_data = np.array(self.p_net.get_node_attrs_data(n_attrs)).T  # (num_nodes, num_attrs)
        # self.node_attr_benchmarks = node_attrs_data.max(axis=0)
        # norm_node_attrs_data = node_attrs_data / self.node_attr_benchmarks
        norm_node_attrs_data = (node_attrs_data - node_attrs_data.min(axis=0)) / (node_attrs_data.max(axis=0) - node_attrs_data.min(axis=0))
        # sum_bw
        e_attrs = self.p_net.get_link_attrs(['resource'])
        link_aggr_attrs_data = np.array(self.p_net.get_aggregation_attrs_data(e_attrs, aggr='sum')).T  # (num_links, num_attrs)
        # self.link_sum_attr_benchmarks = link_aggr_attrs_data.max(axis=0)
        # norm_link_aggr_attrs_data = link_aggr_attrs_data / self.link_sum_attr_benchmarks
        norm_link_aggr_attrs_data = (link_aggr_attrs_data - link_aggr_attrs_data.min(axis=0)) / (link_aggr_attrs_data.max(axis=0) - link_aggr_attrs_data.min(axis=0))
        # degree
        degree_data = np.array([self.p_net_node_degrees]).T
        # # avg_dst
        # if len(self.selected_p_net_nodes) == 0:
        #     avg_distance = np.zeros(self.p_net.num_nodes)
        # else:
        #     avg_distance = []
        #     for u in range(self.p_net.num_nodes):
        #         sum_dst = 0
        #         for v in self.selected_p_net_nodes:
        #             sum_dst += nx.shortest_path_length(self.p_net, source=u, target=v)
        #         sum_dst /= (len(self.selected_p_net_nodes) + 1)
        #         avg_distance.append(sum_dst)
        #     avg_distance = (avg_distance - np.min(avg_distance)) / (np.max(avg_distance) - np.min(avg_distance))
        # avg_distance_data = np.array([avg_distance]).T
        p_net_obs = np.concatenate((norm_node_attrs_data, norm_link_aggr_attrs_data, degree_data), axis=-1)
        return p_net_obs