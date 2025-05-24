import numpy as np
import networkx as nx
from gym import spaces
from omegaconf import open_dict
from virne.network.attribute.attribute_benchmark_manager import AttributeBenchmarkManager
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry
from virne.solver.learning.utils import get_random_unexistent_links, get_unexistent_link_pairs, sort_edge_index


class InstanceRLEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, logger, config, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'
        super(InstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)
        self.init_candidates_dict = self.controller.construct_candidates_dict(self.v_net, self.p_net)
        self.p_net_attribute_benchmarks = AttributeBenchmarkManager.get_from_cache('p_net')
        self.link_attr_benchmarks = self.p_net_attribute_benchmarks.link_attr_benchmarks

    def get_observation(self):
        obs = super(InstanceRLEnv, self).get_observation()
        v_net_obs = {
            'edge_index': obs['v_net_edge_index'],
            'edge_attr': obs['v_net_edge_attr'],
        }
        p_net_obs = {
            'edge_index': obs['p_net_edge_index'],
            'edge_attr': obs['p_net_edge_attr'],
        }
        vp_obs = self._get_vp_obs()
        v_augumented_obs, p_augumented_obs = self.get_augumented_observation(v_net_obs, p_net_obs)
        return {**obs, **vp_obs, **v_augumented_obs, **p_augumented_obs}

    def _get_vp_obs(self):
        # mapping edge
        vp_mapping_edge_index = []
        vp_mapping_edge_attr = []
        for v_node_id, p_node_id in self.solution['node_slots'].items():
            vp_mapping_edge_index.append([v_node_id, p_node_id])
            vp_mapping_edge_attr.append([1.])
        vp_mapping_edge_index = np.array(vp_mapping_edge_index).astype(np.int64).reshape(-1, 2).T
        vp_mapping_edge_attr = np.array(vp_mapping_edge_attr) if len(vp_mapping_edge_attr) != 0 else np.array([[]])
        if vp_mapping_edge_attr.shape == (1, 0):
            vp_mapping_edge_attr = vp_mapping_edge_attr.T
        
        # if len(vp_mapping_edge_index) != 0:
            # vp_mapping_edge_index = np.concatenate([vp_mapping_edge_index, vp_mapping_edge_index[:, [1,0]]], axis=0).T
            # vp_mapping_edge_attr = np.concatenate([vp_mapping_edge_attr, vp_mapping_edge_attr], axis=0)
        # imaginary edge
        vp_imaginary_edge_index = []
        vp_imaginary_edge_attr = []
        for p_node_id in self.init_candidates_dict[self.curr_v_node_id]:
            if p_node_id not in self.selected_p_net_nodes:
                vp_imaginary_edge_index.append([self.curr_v_node_id, p_node_id])
                vp_imaginary_edge_attr.append([1.])
        vp_imaginary_edge_index = np.array(vp_imaginary_edge_index).astype(np.int64).reshape(-1, 2).T
        vp_imaginary_edge_attr = np.array(vp_imaginary_edge_attr) if len(vp_imaginary_edge_attr) != 0 else np.array([[]])
        if vp_imaginary_edge_attr.shape == (1, 0):
            vp_imaginary_edge_attr = vp_imaginary_edge_attr.T
        # if len(vp_imaginary_edge_index) != 0:
            # vp_imaginary_edge_index = np.concatenate([vp_imaginary_edge_index, vp_imaginary_edge_index[:, [1,0]]], axis=0).T
            # vp_imaginary_edge_attr = np.concatenate([vp_imaginary_edge_attr, vp_imaginary_edge_attr], axis=0)
        vp_obs = {
            'vp_mapping_edge_index': vp_mapping_edge_index,
            'vp_mapping_edge_attr': vp_mapping_edge_attr,
            'vp_imaginary_edge_index': vp_imaginary_edge_index,
            'vp_imaginary_edge_attr': vp_imaginary_edge_attr,
        }
        return vp_obs

    def get_augumented_observation(self, v_net_obs, p_net_obs):
        # v_num_added_links = int(self.v_net.num_edges / 2)
        v_num_added_links = min(self.v_net.num_nodes, self.v_net.num_edges)
        v_non_existence_link_pairs = get_unexistent_link_pairs(self.v_net)
        v_added_link_pairs = get_random_unexistent_links(v_non_existence_link_pairs, v_num_added_links)
        v_net_aug_edge_index = np.concatenate((v_net_obs['edge_index'], v_added_link_pairs.T), axis=-1)
        v_net_aug_edge_attr = np.concatenate((v_net_obs['edge_attr'], np.zeros((v_added_link_pairs.shape[0], 1))), axis=0)
        v_net_aug_edge_index, v_net_aug_edge_attr = sort_edge_index(v_net_aug_edge_index, v_net_aug_edge_attr, num_nodes=self.v_net.num_nodes, sort_by_row=True)
        v_augumented_obs = {
            'v_net_aug_edge_index': v_net_aug_edge_index,
            'v_net_aug_edge_attr': v_net_aug_edge_attr,
        }
        v_bw_resource = nx.get_edge_attributes(self.v_net, 'bw')
        unrouted_link_list = list(set(self.v_net.edges()) - set(self.solution.link_paths.keys()))
        unrouted_link_resource_list = [v_bw_resource[link] for link in unrouted_link_list]
        min_unrouted_link_resource = min(unrouted_link_resource_list) if len(unrouted_link_resource_list) !=0 else 0
        # p_num_added_links = int(self.p_net.num_edges / 2)
        p_num_added_links = self.p_net.num_nodes
        # p_num_added_links = self.p_net.num_edges * 2
        p_non_existence_link_pairs = get_unexistent_link_pairs(self.p_net)
        p_added_link_pairs = get_random_unexistent_links(p_non_existence_link_pairs, p_num_added_links)
        p_net_aug_edge_index = np.concatenate((p_net_obs['edge_index'], p_added_link_pairs.T), axis=-1)
        p_net_added_edge_attr = np.ones((p_added_link_pairs.shape[0], 1)) * (min_unrouted_link_resource - 1) / self.link_attr_benchmarks['bw']
        p_net_aug_edge_attr = np.concatenate((p_net_obs['edge_attr'], p_net_added_edge_attr), axis=0)
        p_net_aug_edge_index, p_net_aug_edge_attr = sort_edge_index(p_net_aug_edge_index, p_net_aug_edge_attr, num_nodes=self.p_net.num_nodes, sort_by_row=True)
        p_augumented_obs = {
            'p_net_aug_edge_index': p_net_aug_edge_index,
            'p_net_aug_edge_attr': p_net_aug_edge_attr,
        }
        return v_augumented_obs, p_augumented_obs
