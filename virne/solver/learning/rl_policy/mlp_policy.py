
import torch.nn as nn
import torch.nn.functional as F
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry
from virne.solver.learning.neural_network import MLPNet, GCNConvNet, GATConvNet, DeepEdgeFeatureGAT


@ActorCriticRegistry.register('mlp')
class MlpActorCritic(BaseActorCritic):
    
    def __init__(self, feature_dim, action_dim, num_layers=3, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(MlpActorCritic, self).__init__()
        self.actor = Actor(feature_dim, action_dim, num_layers, embedding_dim, dropout_prob, batch_norm)
        self.critic = Critic(feature_dim, action_dim, num_layers, embedding_dim, dropout_prob, batch_norm)


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, num_layers=3, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(Actor, self).__init__()
        self.net = MLPNet(feature_dim, 1, num_layers, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)

    def forward(self, obs):
        """Return logits of actions"""
        action_logits = self.net(obs['p_net_x'])
        action_logits = action_logits.squeeze(-1)
        return action_logits


class Critic(nn.Module):
    def __init__(self, feature_dim, action_dim, num_layers=3, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(Critic, self).__init__()
        self.net = MLPNet(feature_dim, 1, num_layers, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        # self.net = nn.Sequential(
        #     nn.Linear(feature_dim, 1),
        #     nn.Flatten(),
        #     nn.ReLU(),
        #     nn.Linear(action_dim, 1)
        # )
        
    def forward(self, obs):
        """Return logits of actions"""
        values = self.net(obs['p_net_x'])
        values = values.mean(dim=1)
        return values
