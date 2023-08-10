import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch

from virne.solver import registry
from .instance_env import InstanceRLEnv
from .net import ActorCritic
from virne.solver.learning.rl_base import RLSolver, PPOSolver, A2CSolver, InstanceAgent, A3CSolver
from ..utils import get_pyg_data
from ..obs_handler import POSITIONAL_EMBEDDING_DIM


@registry.register(
    solver_name='ddpg_attention',
    solver_type='r_learning')
class DdpgAttentionSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self, InstanceRLEnv)
        PPOSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)

def make_policy(agent, **kwargs):
    num_vn_attrs = agent.v_sim_setting_num_node_resource_attrs
    num_vl_attrs = agent.v_sim_setting_num_link_resource_attrs
    policy = ActorCritic(p_net_num_nodes=agent.p_net_setting_num_nodes, 
                        p_net_feature_dim=num_vn_attrs*2 + num_vl_attrs*2 + 1, 
                        v_node_feature_dim=num_vn_attrs+num_vl_attrs+1,
                        embedding_dim=agent.embedding_dim, 
                        dropout_prob=agent.dropout_prob, 
                        batch_norm=agent.batch_norm).to(agent.device)
    optimizer = torch.optim.Adam([
            {'params': policy.actor.parameters(), 'lr': agent.lr_actor},
            {'params': policy.critic.parameters(), 'lr': agent.lr_critic},
        ], weight_decay=agent.weight_decay
    )
    return policy, optimizer


def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        tensor_obs_p_net_x = torch.FloatTensor(np.array([obs['p_net_x']])).to(device)
        tensor_obs_v_net_x = torch.FloatTensor(np.array([obs['v_net_x']])).to(device)
        tensor_obs_curr_v_node_id = torch.LongTensor(np.array([obs['curr_v_node_id']])).to(device)
        tensor_obs_action_mask = torch.FloatTensor(np.array([obs['action_mask']])).to(device)
        tensor_obs_v_net_size = torch.FloatTensor(np.array([obs['v_net_size']])).to(device)
        return {'p_net_x': tensor_obs_p_net_x, 'v_net_x': tensor_obs_v_net_x, 'curr_v_node_id': tensor_obs_curr_v_node_id, 'action_mask': tensor_obs_action_mask, 'v_net_size': tensor_obs_v_net_size}
    # batch
    elif isinstance(obs, list):
        p_net_x_list, v_net_x_list, v_net_size_list, curr_v_node_id_list, action_mask_list = [], [], [], [], []
        for observation in obs:
            p_net_x_list.append(observation['p_net_x'])
            v_net_x_list.append(observation['v_net_x'])
            v_net_size_list.append(observation['v_net_size'])
            curr_v_node_id_list.append(observation['curr_v_node_id'])
            action_mask_list.append(observation['action_mask'])
        tensor_obs_p_net_x = torch.FloatTensor(np.array(p_net_x_list)).to(device)
        tensor_obs_v_net_x = torch.FloatTensor(np.array(v_net_x_list)).to(device)
        tensor_obs_v_net_size = torch.FloatTensor(np.array(v_net_size_list)).to(device)
        tensor_obs_curr_v_node_id = torch.LongTensor(np.array(curr_v_node_id_list)).to(device)
        tensor_obs_action_mask = torch.FloatTensor(np.array(action_mask_list)).to(device)
        return {'p_net_x': tensor_obs_p_net_x, 'v_net_x': tensor_obs_v_net_x, 'v_net_size': tensor_obs_v_net_size, 'curr_v_node_id': tensor_obs_curr_v_node_id, 'action_mask': tensor_obs_action_mask}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")