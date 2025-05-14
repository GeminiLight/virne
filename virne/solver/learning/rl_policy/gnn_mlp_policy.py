import torch
import torch.nn as nn
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry
from virne.solver.learning.neural_network import MLPNet, GCNConvNet, GATConvNet, DeepEdgeFeatureGAT


def _make_actor_critic(gnn_class):
    class ActorCritic(BaseActorCritic):
        def __init__(self, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=3, embedding_dim=128, p_net_edge_dim=None, dropout_prob=0., batch_norm=False, **kwargs):
            super().__init__()
            self.actor = Actor(gnn_class, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers, embedding_dim, p_net_edge_dim, dropout_prob, batch_norm)
            self.critic = Critic(gnn_class, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers, embedding_dim, p_net_edge_dim, dropout_prob, batch_norm)
    return ActorCritic

@ActorCriticRegistry.register('gcn_mlp')
class GcnMlpActorCritic(_make_actor_critic(GCNConvNet)): pass

@ActorCriticRegistry.register('gat_mlp')
class GatMlpActorCritic(_make_actor_critic(GATConvNet)): pass

@ActorCriticRegistry.register('deep_edge_gat_mlp')
class DeepEdgeFeatureGATActorCritic(_make_actor_critic(DeepEdgeFeatureGAT)): pass

class Actor(nn.Module):
    def __init__(self, gnn_class, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=3, embedding_dim=128, p_net_edge_dim=None, dropout_prob=0., batch_norm=False, **kwargs):
        super().__init__()
        self.encoder = GnnMlpEncoder(gnn_class, p_net_x_dim, v_node_feature_dim, num_layers, embedding_dim, p_net_edge_dim, dropout_prob, batch_norm)
        self.head = MLPNet(embedding_dim, 1, num_layers=2, embedding_dims=int(embedding_dim / 2), batch_norm=batch_norm, dropout_prob=dropout_prob)
    def forward(self, obs):
        fusion = self.encoder(obs['p_net'], obs['v_node_x'])
        return self.head(fusion).squeeze(-1)

class Critic(nn.Module):
    def __init__(self, gnn_class, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, num_layers=3, embedding_dim=128, p_net_edge_dim=None, dropout_prob=0., batch_norm=False, **kwargs):
        super().__init__()
        self.encoder = GnnMlpEncoder(gnn_class, p_net_x_dim, v_node_feature_dim, num_layers, embedding_dim, p_net_edge_dim, dropout_prob, batch_norm)
        self.head = MLPNet(embedding_dim, 1, num_layers=2, embedding_dims=int(embedding_dim / 2), batch_norm=batch_norm, dropout_prob=dropout_prob)
    def forward(self, obs):
        fusion = self.encoder(obs['p_net'], obs['v_node_x'])
        return self.head(fusion).mean(dim=1)

class GnnMlpEncoder(nn.Module):
    def __init__(self, gnn_class, p_net_x_dim, v_node_feature_dim, num_layers=3, embedding_dim=128, p_net_edge_dim=None, dropout_prob=0., batch_norm=False, **kwargs):
        super().__init__()
        self.gnn = gnn_class(p_net_x_dim, embedding_dim, num_layers=num_layers, embedding_dim=embedding_dim, edge_dim=p_net_edge_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, return_batch=True)
        self.mlp = MLPNet(v_node_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
    def forward(self, p_net, v_net_x):
        p_emb = self.gnn(p_net)
        v_emb = self.mlp(v_net_x)
        return p_emb + v_emb.unsqueeze(1).expand(-1, p_emb.shape[1], -1)
