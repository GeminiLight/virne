# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .sub_env import SubEnv
from .net import Actor, ActorCritic, Critic
from ..rl_solver import *
from base import Solution, SolutionStepEnvironment
from solver import registry


@registry.register(
    solver_name='pg_mlp',
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning')
class PgMlpSolver(PPOSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Multilayer Perceptron (MLP) as the neural network model.
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        super(PgMlpSolver, self).__init__(controller, recorder, counter, **kwargs)
        action_dim = kwargs['p_net_setting']['num_nodes']
        feature_dim = 4 * action_dim  # (n_attrs, e_attrs, dist, degree)
        self.policy = ActorCritic(feature_dim, action_dim, self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_actor},
        ])
        self.SubEnv = SubEnv
        self.preprocess_obs = obs_as_tensor
        
    def preprocess_obs(self, observation):
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(self.device)
        return observation

    def preprocess_batch_obs(self, obs_batch):
        """Preprocess the observation to adapte to batch mode."""
        observation = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        return observation


def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, list):
        obs_batch = obs
        """Preprocess the observation to adapte to batch mode."""
        observation = torch.FloatTensor(np.array(obs_batch)).to(device)
        return observation
    # batch
    else:
        observation = obs
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(device)
        return observation