import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(feature_dim, action_dim, embedding_dim)
        self.critic = Critic(feature_dim, action_dim, embedding_dim)

    def act(self, x):
        return self.actor(x)

    def evaluate(self, obs):
        """Return logits of actions"""
        values = self.critic(obs)
        return values


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
        action_logits = self.net(obs)
        # action_logits = self.relu(out)
        return action_logits

    def act(self, obs):
        return self(obs)


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
        values = self.net(obs)
        return values