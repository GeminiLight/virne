# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch.nn as nn
import torch.nn.functional as F
from .base_policy import BaseActorCritic
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry
from virne.solver.learning.neural_network import ResNetBlock


@ActorCriticRegistry.register('cnn')
class CnnActorCritic(BaseActorCritic):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=128, **kwargs):
        super(CnnActorCritic, self).__init__()
        self.actor = Actor(feature_dim, action_dim, embedding_dim, **kwargs)
        self.critic = Critic(feature_dim, action_dim, embedding_dim, **kwargs)


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, embedding_dim=128, **kwargs):
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
    def __init__(self, feature_dim, action_dim, embedding_dim=128, **kwargs):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[1, feature_dim], stride=[1, 1]),
            nn.Flatten(),
            nn.ReLU(),
        )
        
    def forward(self, obs):
        """Return logits of actions"""
        x = obs['p_net_x']
        values = self.net(x)
        values = values.mean(dim=1)
        return values