# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .instance_env import InstanceEnv
from .net import Actor, ActorCritic, Critic
from virne.solver.learning.rl_base import *
from virne.base import Solution, SolutionStepEnvironment
from virne.solver import registry


@registry.register(
    solver_name='pg_cnn', 
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning')
class PgCnnSolver(InstanceAgent, PPOSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Convolutional Neural Network (CNN) as the neural network model.
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self, InstanceEnv)
        PPOSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        self.use_negative_sample = False

def make_policy(agent, **kwargs):
    action_dim = agent.p_net_setting_num_nodes
    feature_dim = agent.p_net_setting_num_node_resource_attrs + agent.p_net_setting_num_link_resource_attrs + 2  # (n_attrs, e_attrs, dist, degree)
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


# def obs_as_tensor(obs, device):
#     # one
#     if isinstance(obs, list):
#         obs_batch = obs
#         """Preprocess the observation to adapte to batch mode."""
#         observation = torch.FloatTensor(np.array(obs_batch)).to(device)
#         return observation
#     # batch
#     else:
#         observation = obs
#         observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(device)
#         return observation