# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from virne.solver import registry
from .instance_env import InstanceEnv
from .net import Actor, ActorCritic, Critic
from virne.solver.learning.rl_base import *
from virne.base import Solution, SolutionStepEnvironment



@registry.register(
    solver_name='pg_cnn2', 
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning')
class PgCnn2Solver(InstanceAgent, PPOSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Convolutional Neural Network (CNN) as the neural network model.
    Additionally, more graph features are used as the input of the CNN.
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        kwargs['use_negative_sample'] = False
        InstanceAgent.__init__(self, InstanceEnv)
        PPOSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        self.use_negative_sample = False


def make_policy(agent, **kwargs):
    action_dim = agent.p_net_setting_num_nodes
    feature_dim = agent.p_net_setting_num_node_resource_attrs + agent.p_net_setting_num_link_resource_attrs + 5
    policy = ActorCritic(feature_dim, action_dim, agent.embedding_dim).to(agent.device)
    optimizer = torch.optim.Adam([
        {'params': policy.parameters(), 'lr': agent.lr_actor}
    ], weight_decay=agent.weight_decay)
    return policy, optimizer

def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        obs_batch = obs
        p_net_x = torch.FloatTensor(np.array([obs_batch['p_net_x']])).to(device)
        action_mask = torch.FloatTensor(np.array([obs_batch['action_mask']])).to(device)
        tensor_obs = {'p_net_x': p_net_x, 'action_mask': action_mask}
        return tensor_obs
    # batch
    else:
        p_net_x = torch.FloatTensor(np.array([observation['p_net_x'] for observation in obs])).to(device)
        action_mask = torch.FloatTensor(np.array([observation['action_mask'] for observation in obs])).to(device)
        tensor_obs = {'p_net_x': p_net_x, 'action_mask': action_mask}
        return tensor_obs
