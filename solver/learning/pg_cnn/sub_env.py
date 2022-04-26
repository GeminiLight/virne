import numpy as np
import networkx as nx
from gym import spaces
from ..sub_rl_environment import JointPRStepSubRLEnv, PlaceStepSubRLEnv


class SubEnv(PlaceStepSubRLEnv):

    def __init__(self, pn, vn):
        super(SubEnv, self).__init__(pn, vn)
        num_pn_node_attrs = len(self.pn.get_node_attrs(['resource']))
        num_pn_edge_attrs = len(self.pn.get_edge_attrs(['resource']))
        num_obs_attrs = num_pn_node_attrs + num_pn_edge_attrs + 2
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.pn.num_nodes, num_obs_attrs, 1), dtype=np.float32)

    def compute_reward(self, solution):
        r"""Calculate deserved reward according to the result of taking action."""
        if solution['result'] :
            reward = solution['vn_rc_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = 0
        else:
            reward = 0
        return reward

    def get_observation(self):
        pn_obs = self.get_pn_obs()
        return self.obs_for_cnn(pn_obs)
    
    def get_pn_obs(self):
        # (cpu_remain, degree, sum_bw, avg_dst)
        # resource
        n_attrs = self.pn.get_node_attrs(['resource'])
        node_attrs_data = np.array(self.pn.get_node_attrs_data(n_attrs)).T  # (num_nodes, num_attrs)
        # self.node_attrs_benchmark = node_attrs_data.max(axis=0)
        # norm_node_attrs_data = node_attrs_data / self.node_attrs_benchmark
        norm_node_attrs_data = (node_attrs_data - node_attrs_data.min(axis=0)) / (node_attrs_data.max(axis=0) - node_attrs_data.min(axis=0))
        # sum_bw
        e_attrs = self.pn.get_edge_attrs(['resource'])
        edge_aggr_attrs_data = np.array(self.pn.get_aggregation_attrs_data(e_attrs, aggr='sum')).T  # (num_edges, num_attrs)
        # self.edge_aggr_attrs_benchmark = edge_aggr_attrs_data.max(axis=0)
        # norm_edge_aggr_attrs_data = edge_aggr_attrs_data / self.edge_aggr_attrs_benchmark
        norm_edge_aggr_attrs_data = (edge_aggr_attrs_data - edge_aggr_attrs_data.min(axis=0)) / (edge_aggr_attrs_data.max(axis=0) - edge_aggr_attrs_data.min(axis=0))
        # avg_dst
        avg_distance = self.obs_handler.get_average_distance(self.pn, self.selected_pn_nodes, normalization=True)
        pn_obs = np.concatenate((norm_node_attrs_data, norm_edge_aggr_attrs_data, self.pn_node_degrees, avg_distance), axis=-1)
        return pn_obs

    def obs_for_cnn(self, obs):
        return np.expand_dims(obs, axis=0)