
import torch.nn as nn
import torch.nn.functional as F
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry
from virne.solver.learning.neural_network import *


@ActorCriticRegistry.register('att')
class AttActorCritic(BaseActorCritic):
    
    def __init__(self, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=3, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(AttActorCritic, self).__init__()
        self.actor = Actor(p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers, embedding_dim, dropout_prob, batch_norm)
        self.critic = Critic(p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers, embedding_dim, dropout_prob, batch_norm)


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=3, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(Actor, self).__init__()
        self.encoder = AttEncoder(p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=num_layers, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.lin_fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, obs):
        """Return logits of actions"""
        fusion_embeddings = self.encoder(obs)
        action_logits = self.lin_fusion(fusion_embeddings).squeeze(-1)
        return action_logits


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=3, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(Critic, self).__init__()
        self.encoder = AttEncoder(p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=num_layers, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)

    def forward(self, obs):
        """Return logits of actions"""
        fusion_embeddings = self.encoder(obs)
        value = fusion_embeddings.mean(dim=1).mean(dim=1).unsqueeze(-1)
        return value



class AttEncoder(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=3, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(AttEncoder, self).__init__()
        self.num_att_heads = kwargs.get('num_att_heads', 1)
        self.p_mlp = MLPNet(p_net_x_dim, embedding_dim, num_layers=num_layers, embedding_dims=None, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.v_mlp = MLPNet(v_node_feature_dim, embedding_dim, num_layers=num_layers, embedding_dims=None, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.att = nn.MultiheadAttention(embedding_dim, num_heads=self.num_att_heads, batch_first=True)

    def forward(self, obs):
        """Return logits of actions"""
        p_node_embeddings = self.p_mlp(obs['p_net_x'])
        v_node_embedding = self.v_mlp(obs['v_node_x'])
        init_fusion_embeddings = p_node_embeddings + v_node_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1)
        att_fusion_embeddings, _ = self.att(init_fusion_embeddings, init_fusion_embeddings, init_fusion_embeddings)
        fusion_embeddings = att_fusion_embeddings + init_fusion_embeddings
        return fusion_embeddings
    