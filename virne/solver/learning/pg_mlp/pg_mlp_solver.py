# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from virne.solver.learning.rl_base import InstanceAgent, PGSolver
from virne.base import Solution
from virne.solver import registry
from .instance_env import InstanceRLEnv
from .net import Actor, ActorCritic, Critic


@registry.register(
    solver_name='pg_mlp',
    solver_type='r_learning')
class PgMlpSolver(InstanceAgent, PGSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Multilayer Perceptron (MLP) as the neural network model.
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self, InstanceRLEnv)
        PGSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        
    def preprocess_obs(self, observation):
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(self.device)
        return observation

    def preprocess_batch_obs(self, obs_batch):
        """Preprocess the observation to adapt to batch mode."""
        observation = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        return observation

def make_policy(agent, **kwargs):
    action_dim = agent.p_net_setting_num_nodes
    feature_dim = 4 * action_dim  # (n_attrs, e_attrs, dist, degree)
    policy = ActorCritic(feature_dim, action_dim, agent.embedding_dim).to(agent.device)
    optimizer = torch.optim.Adam([
            {'params': policy.actor.parameters(), 'lr': agent.lr_actor},
            {'params': policy.critic.parameters(), 'lr': agent.lr_critic},
        ], weight_decay=agent.weight_decay
    )
    return policy, optimizer


def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, list):
        obs_batch = obs
        """Preprocess the observation to adapt to batch mode."""
        observation = torch.FloatTensor(np.array(obs_batch)).to(device)
        return observation
    # batch
    else:
        observation = obs
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(device)
        return observation