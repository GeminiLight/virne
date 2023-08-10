import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch

from ..neural_network import *


class ActorCritic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim=64, dropout_prob=0., batch_norm=False):
        super(ActorCritic, self).__init__()
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)

    def act(self, obs):
        return self.actor(obs)

    def evaluate(self, obs):
        return self.critic(obs)


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim=64, dropout_prob=0., batch_norm=False):
        super(Actor, self).__init__()
        self.p_mlp = MLPNet(p_net_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.v_mlp = MLPNet(v_node_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.att = nn.MultiheadAttention(embedding_dim, num_heads=1, batch_first=True)
        self.lin_fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, obs):
        """Return logits of actions"""
        p_node_embeddings = self.p_mlp(obs['p_net_x'])
        v_node_embedding = self.v_mlp(obs['v_net_x'])
        init_fusion_embeddings = p_node_embeddings + v_node_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1)
        att_fusion_embeddings, _ = self.att(init_fusion_embeddings, init_fusion_embeddings, init_fusion_embeddings)
        fusion_embeddings = att_fusion_embeddings + init_fusion_embeddings
        action_logits = self.lin_fusion(fusion_embeddings).squeeze(-1)
        return action_logits


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim=64, dropout_prob=0., batch_norm=False):
        super(Critic, self).__init__()
        self.p_mlp = MLPNet(p_net_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.v_mlp = MLPNet(v_node_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.att = nn.MultiheadAttention(embedding_dim, num_heads=1, batch_first=True)
        # self.lin_fusion = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 1),
        # )

    def forward(self, obs):
        """Return logits of actions"""
        p_node_embeddings = self.p_mlp(obs['p_net_x'])
        v_node_embedding = self.v_mlp(obs['v_net_x'])
        init_fusion_embeddings = p_node_embeddings + v_node_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1)
        att_fusion_embeddings, _ = self.att(init_fusion_embeddings, init_fusion_embeddings, init_fusion_embeddings)
        fusion_embeddings = att_fusion_embeddings + init_fusion_embeddings
        # action_logits = self.lin_fusion(fusion_embeddings).squeeze(-1)
        # value = action_logits.mean(dim=1)
        value = fusion_embeddings.mean(dim=1).mean(dim=1).unsqueeze(-1)
        return value

