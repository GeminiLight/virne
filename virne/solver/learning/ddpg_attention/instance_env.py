import numpy as np
import networkx as nx
from gym import spaces
from virne.solver.learning.rl_base import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.obs_handler import POSITIONAL_EMBEDDING_DIM, P_NODE_STATUS_DIM, V_NODE_STATUS_DIM


class InstanceRLEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(InstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        # self.calcuate_graph_metrics(degree=True, closeness=False, eigenvector=False, betweenness=False)

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_node_obs = self._get_v_node_obs()
        return {
            'p_net_x': p_net_obs['x'],
            'v_net_x': v_node_obs['x'],
            'curr_v_node_id': self.curr_v_node_id,
            'v_net_size': self.v_net.num_nodes,
            'action_mask': self.generate_action_mask(),
        }

    def _get_p_net_obs(self, ):
        attr_type_list = ['resource', 'extrema']
        v_node_min_link_demend = self.obs_handler.get_v_node_min_link_demend(self.v_net, self.curr_v_node_id)
        # node data
        p_node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=attr_type_list, node_attr_benchmarks=self.node_attr_benchmarks)
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=attr_type_list, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        p_nodes_status = np.zeros((self.p_net.num_nodes, 1), dtype=np.float32)
        selected_p_nodes = list(self.solution['node_slots'].values())
        p_nodes_status[selected_p_nodes, 0] = 1.
        v2p_node_link_demand = self.obs_handler.get_v2p_node_link_demand(self.p_net, self.v_net, self.solution['node_slots'], self.curr_v_node_id, self.link_attr_benchmarks)
        node_data = np.concatenate((p_node_data, p_node_link_sum_resource, p_nodes_status), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        # data
        p_net_obs = {
            'x': node_data,
        }
        return p_net_obs

    def _get_v_node_obs(self):
        if self.curr_v_node_id  >= self.v_net.num_nodes:
            return []
        norm_unplaced = (self.v_net.num_nodes - (self.curr_v_node_id + 1)) / self.v_net.num_nodes

        node_demand = []
        for n_attr in self.v_net.get_node_attrs('resource'):
            node_demand.append(self.v_net.nodes[self.curr_v_node_id][n_attr.name] / self.node_attr_benchmarks[n_attr.name])
        norm_node_demand = np.array(node_demand, dtype=np.float32)

        # max_link_demand = []
        # mean_link_demand = []
        sum_link_demand = []
        num_neighbors = len(self.v_net.adj[self.curr_v_node_id]) / self.v_net.num_nodes
        for l_attr in self.v_net.get_link_attrs('resource'):
            link_demand = [self.v_net.links[(n, self.curr_v_node_id)][l_attr.name] for n in self.v_net.adj[self.curr_v_node_id]]
            # max_link_demand.append(max(link_demand) / self.link_attr_benchmarks[l_attr.name])
            # mean_link_demand.append((sum(link_demand) / len(link_demand)) / self.link_attr_benchmarks[l_attr.name])
            sum_link_demand.append((sum(link_demand)) / self.link_sum_attr_benchmarks[l_attr.name])

        node_data = np.concatenate([norm_node_demand, sum_link_demand, [norm_unplaced]], axis=0)
        return {
            'x': node_data
        }

    def compute_reward(self, solution):
        """Calculate deserved reward according to the result of taking action."""
        weight = (1 / self.v_net.num_nodes)
        if solution['result']:
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = weight
        else:
            reward = - weight
        self.solution['v_net_reward'] += reward
        return reward

