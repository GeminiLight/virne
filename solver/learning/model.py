import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .net import DeepEdgeFeatureGAT, GraphAttentionPooling, MLPNet, PositionalEncoder


class HRLPolicy(nn.Module):

    def __init__(self, p_net_node_dim, p_net_edge_dim, v_net_node_dim, v_net_edge_dim, v_net_attrs_dim, upper_output_dim, lower_output_dim, num_gnn_layers=5, alpha=0.2, theta=0.2, embedding_dim=128, hidden_dim=None, num_heads=1, batch_norm=False, dropout_prob=0.25):
        super(HRLPolicy, self).__init__()
        self.upper = UpperActorCritic(p_net_node_dim, p_net_edge_dim, v_net_node_dim, v_net_edge_dim, upper_output_dim, v_net_attrs_dim, num_gnn_layers=num_gnn_layers, alpha=alpha, theta=theta, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.lower = LowerActorCritic(p_net_node_dim, p_net_edge_dim, v_net_node_dim, v_net_edge_dim, lower_output_dim, v_net_attrs_dim, num_gnn_layers=num_gnn_layers, alpha=alpha, theta=theta, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, batch_norm=batch_norm, dropout_prob=dropout_prob)


class UpperActorCritic(nn.Module):

    def __init__(self, p_net_node_dim, p_net_edge_dim, v_net_node_dim, v_net_edge_dim, v_net_attrs_dim, output_dim, num_gnn_layers=5, alpha=0.2, theta=0.2, embedding_dim=128, hidden_dim=None, num_heads=1, batch_norm=False, dropout_prob=0.25):
        super(UpperActorCritic, self).__init__()
        self.basic = UpperBasicNet(p_net_node_dim, p_net_edge_dim, v_net_node_dim, v_net_edge_dim, v_net_attrs_dim, num_gnn_layers=num_gnn_layers, alpha=alpha, theta=theta, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.actor = MLPNet(embedding_dim, output_dim, num_layers=3, embedding_dims=[int(embedding_dim / 2), int(embedding_dim / 4)], batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.critic = MLPNet(embedding_dim, 1, num_layers=3, embedding_dims=[int(embedding_dim / 2), int(embedding_dim / 4)], batch_norm=batch_norm, dropout_prob=dropout_prob)

    def act(self, obs):
        p_net_data, v_net_data, v_net_attrs, hidden = obs['p_net_data'], obs['v_net_data'], obs['v_net_attrs'], obs['hidden']
        hidden = self.basic(p_net_data, v_net_data, v_net_attrs, hidden)
        return self.actor(hidden), hidden

    def estimate(self, obs):
        p_net_data, v_net_data, v_net_attrs, hidden = obs['p_net_data'], obs['v_net_data'], obs['v_net_attrs'], obs['hidden']
        hidden = self.basic(p_net_data, v_net_data, v_net_attrs, hidden)
        return self.critic(hidden)


class UpperBasicNet(nn.Module):

    def __init__(self, p_net_node_dim, p_net_edge_dim, v_net_node_dim, v_net_edge_dim, v_net_attrs_dim, num_gnn_layers=5, alpha=0.2, theta=0.2, embedding_dim=128, hidden_dim=None, num_heads=1, batch_norm=False, dropout_prob=0.25):
        super(UpperBasicNet, self).__init__()
        if hidden_dim is None: hidden_dim = embedding_dim
        self.gru_cell = nn.GRUCell(embedding_dim, hidden_dim)
        self.gnn_p_net = DeepEdgeFeatureGAT(p_net_node_dim, embedding_dim, edge_dim=p_net_edge_dim, num_layers=num_gnn_layers, num_heads=num_heads, alpha=alpha, theta=theta, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.gnn_v_net = DeepEdgeFeatureGAT(v_net_node_dim, embedding_dim, edge_dim=v_net_edge_dim, num_layers=num_gnn_layers, num_heads=num_heads, alpha=alpha, theta=theta, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.gap_p_net = GraphAttentionPooling(embedding_dim)
        self.gap_v_net = GraphAttentionPooling(embedding_dim)
        mlp_embedding_dims = [embedding_dim * 2, embedding_dim]
        self.mlp_v_net_attrs = MLPNet(v_net_attrs_dim, embedding_dim, num_layers=2, embedding_dims=[embedding_dim], batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.mlp_fusion = MLPNet(embedding_dim * 3, embedding_dim, num_layers=3, embedding_dims=mlp_embedding_dims, batch_norm=batch_norm, dropout_prob=dropout_prob)

    def forward(self, p_net, v_net, v_net_attrs, hidden):
        p_net_graph_embedding = self.gap_p_net(self.gnn_p_net(p_net), p_net.batch)
        v_net_graph_embedding = self.gap_v_net(self.gnn_v_net(v_net), v_net.batch)
        v_net_attrs_embedding = self.mlp_v_net_attrs(v_net_attrs)
        # fusion
        fusion = torch.concat([p_net_graph_embedding, v_net_graph_embedding, v_net_attrs_embedding], dim=-1)
        fusion_embedding = self.mlp_fusion(fusion)
        # gru
        hidden = self.gru_cell(fusion_embedding, hidden)
        return hidden


class LowerActorCritic(nn.Module):

    def __init__(self, p_net_node_dim, p_net_edge_dim, v_net_node_dim, v_net_attrs_dim, v_net_edge_dim, output_dim, num_gnn_layers=5, alpha=0.2, theta=0.2, embedding_dim=128, hidden_dim=None, num_heads=1, batch_norm=False, dropout_prob=0.25):
        super(LowerActorCritic, self).__init__()
        self.encoder = Encoder(v_net_node_dim, v_net_edge_dim, num_gnn_layers=num_gnn_layers, alpha=alpha, theta=theta, embedding_dim=embedding_dim, num_heads=num_heads, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.basic = LowerBasicNet(p_net_node_dim, p_net_edge_dim, v_net_attrs_dim, num_gnn_layers=num_gnn_layers, alpha=alpha, theta=theta, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.actor = MLPNet(embedding_dim, output_dim, num_layers=3, embedding_dims=[int(embedding_dim * 2), int(embedding_dim * 2)], batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.critic = MLPNet(embedding_dim, 1, num_layers=3, embedding_dims=[int(embedding_dim / 2), int(embedding_dim / 4)], batch_norm=batch_norm, dropout_prob=dropout_prob)

    def act(self, obs):
        p_net_data, v_node_embedding, v_net_attrs, hidden = obs['p_net_data'], obs['v_node_embedding'], obs['v_net_attrs'], obs['hidden']
        hidden = self.basic(p_net_data, v_node_embedding, v_net_attrs, hidden)
        return self.actor(hidden), hidden

    def estimate(self, obs):
        p_net_data, v_node_embedding, v_net_attrs, hidden = obs['p_net_data'], obs['v_node_embedding'], obs['v_net_attrs'], obs['hidden']
        hidden = self.basic(p_net_data, v_node_embedding, v_net_attrs, hidden)
        return self.critic(hidden)


class LowerBasicNet(nn.Module):

    def __init__(self, p_net_node_dim, p_net_edge_dim, v_net_attrs_dim, num_gnn_layers=5, alpha=0.2, theta=0.2, embedding_dim=128, hidden_dim=None, num_heads=1, batch_norm=False, dropout_prob=0.25):
        super(LowerBasicNet, self).__init__()
        if hidden_dim is None: hidden_dim = embedding_dim
        self.gru_cell = nn.GRUCell(embedding_dim, hidden_dim)
        self.gnn_p_net = DeepEdgeFeatureGAT(p_net_node_dim, embedding_dim, edge_dim=p_net_edge_dim, num_layers=num_gnn_layers, num_heads=num_heads, alpha=alpha, theta=theta, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.gap_p_net = GraphAttentionPooling(embedding_dim)
        self.mlp_v_net_attrs = MLPNet(v_net_attrs_dim, embedding_dim, num_layers=2, embedding_dims=[embedding_dim], batch_norm=batch_norm, dropout_prob=dropout_prob)
        mlp_embedding_dims = [embedding_dim * 2, embedding_dim]
        self.mlp_fusion = MLPNet(embedding_dim * 3, embedding_dim, num_layers=3, embedding_dims=mlp_embedding_dims, batch_norm=batch_norm, dropout_prob=dropout_prob)

    def forward(self, p_net, v_node_embedding, v_net_attrs, hidden):
        p_net_graph_embedding = self.gap_p_net(self.gnn_p_net(p_net), p_net.batch)
        v_net_attrs_embedding = self.mlp_v_net_attrs(v_net_attrs)
        # fusion
        fusion = torch.concat([p_net_graph_embedding, v_node_embedding, v_net_attrs_embedding], dim=-1)
        fusion_embedding = self.mlp_fusion(fusion)
        # gru
        hidden = self.gru_cell(fusion_embedding, hidden)
        return hidden


class Encoder(nn.Module):

    def __init__(self, v_net_node_dim, v_net_edge_dim, num_gnn_layers=5, alpha=0.2, theta=0.2, embedding_dim=128, num_heads=1, batch_norm=False, dropout_prob=0.25):
        super(Encoder, self).__init__()
        self.gnn_encoder = DeepEdgeFeatureGAT(v_net_node_dim, embedding_dim, edge_dim=v_net_edge_dim, num_layers=num_gnn_layers, num_heads=num_heads, alpha=alpha, theta=theta, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.pe_encoder = PositionalEncoder(embedding_dim, dropout_prob=dropout_prob)

    def forward(self, v_net):
        v_net_node_embeddings = self.gnn_encoder(v_net)
        v_net_node_embeddings, mask = to_dense_batch(v_net_node_embeddings, v_net.batch)
        v_net_node_embeddings = self.pe_encoder(v_net_node_embeddings)
        return v_net_node_embeddings


if __name__ == '__main__':
    policy = HRLPolicy(p_net_node_dim=7, p_net_edge_dim=2, v_net_node_dim=3, v_net_edge_dim=1, v_net_attrs_dim=2, upper_output_dim=2, lower_output_dim=100)