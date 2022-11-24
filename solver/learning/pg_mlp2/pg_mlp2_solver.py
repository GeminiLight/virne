import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .sub_env import SubEnv
from .net import Actor, ActorCritic, Critic
from ..rl_solver import *


class PGMLP2Solver(PPOSolver):

    name = 'pg_mlp2'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(PGMLP2Solver, self).__init__(controller, recorder, counter, **kwargs)
        num_p_net_node_features = kwargs.get('num_p_net_node_attrs', 2)
        num_p_net_edge_features = kwargs.get('num_p_net_link_attrs', 2)
        feature_dim = 5 + int((num_p_net_node_features + num_p_net_edge_features) / 2)  # (n_attrs, e_attrs, dist, degree)        
        action_dim = kwargs['p_net_setting']['num_nodes']
        self.policy = ActorCritic(feature_dim, action_dim, self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_actor},
        ])
        self.SubEnv = SubEnv
        self.preprocess_obs = obs_as_tensor
        

def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, list):
        obs_batch = obs
        r"""Preprocess the observation to adapte to batch mode."""
        observation = torch.FloatTensor(np.array(obs_batch)).to(device)
        return observation
    # batch
    else:
        observation = obs
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(device)
        return observation
