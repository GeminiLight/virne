# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch.nn as nn
import torch.nn.functional as F

from ..neural_network import ResNetBlock
from virne.solver.learning.rl_base.policy_base import ActorCriticBase


class ActorCritic(ActorCriticBase):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(feature_dim, action_dim, embedding_dim)
        self.critic = Critic(feature_dim, action_dim, embedding_dim)


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[1, feature_dim], stride=[1, 1]),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, obs):
        """Return logits of actions"""
        x = obs['p_net_x']
        action_logits = self.net(x)
        return action_logits


class Critic(nn.Module):
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[1, feature_dim], stride=[1, 1]),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(action_dim, 1)
        )
        
    def forward(self, obs):
        """Return logits of actions"""
        x = obs['p_net_x']
        values = self.net(x)
        return values