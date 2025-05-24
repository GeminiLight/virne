import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry
from virne.solver.learning.neural_network import *


def _make_actor_critic(gnn_class):
    class ActorCritic(BaseActorCritic):
        def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, num_layers=3, **kwargs):
            super().__init__()
            self.actor = Actor(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, num_layers=num_layers, gnn_class=gnn_class, **kwargs)
            self.critic = Critic(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, num_layers=num_layers, gnn_class=gnn_class, **kwargs)
    return ActorCritic

def _make_advanced_actor_critic(gnn_class, use_cost_critic=False, use_lambda_net=False):
    class ActorCritic(BaseActorCritic):
        def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, num_layers=3, **kwargs):
            super().__init__()
            self.actor = Actor(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, num_layers=num_layers, gnn_class=gnn_class, **kwargs)
            self.critic = Critic(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, num_layers=num_layers, gnn_class=gnn_class, **kwargs)
            if use_cost_critic:
                self.cost_critic = Critic(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, num_layers=num_layers, gnn_class=gnn_class)
            if use_lambda_net:
                self.lambda_net = Critic(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, num_layers=num_layers, gnn_class=gnn_class)    
    return ActorCritic


@ActorCriticRegistry.register('dual_gcn')
class BiGcnActorCritic(_make_actor_critic(GCNConvNet)): pass
@ActorCriticRegistry.register('dual_gat')
class BiGatActorCritic(_make_actor_critic(GATConvNet)): pass
@ActorCriticRegistry.register('dual_deep_edge_gat')
class BiDeepEdgeFeatureGatActorCritic(_make_actor_critic(DeepEdgeFeatureGAT)): pass

@ActorCriticRegistry.register('dual_gcn_with_cost')
class BiGcnWithCostActorCritic(_make_advanced_actor_critic(GCNConvNet, use_cost_critic=True)): pass
@ActorCriticRegistry.register('dual_gat_with_cost')
class BiGatWithCostActorCritic(_make_advanced_actor_critic(GATConvNet, use_cost_critic=True)): pass
@ActorCriticRegistry.register('dual_deep_edge_gat_with_cost')
class BiDeepEdgeFeatureGatWithCostActorCritic(_make_advanced_actor_critic(DeepEdgeFeatureGAT, use_cost_critic=True)): pass

@ActorCriticRegistry.register('dual_gcn_with_cost_and_lambda')
class BiGcnWithCostAndLambdaActorCritic(_make_advanced_actor_critic(GCNConvNet, use_cost_critic=True, use_lambda_net=True)): pass
@ActorCriticRegistry.register('dual_gat_with_cost_and_lambda')
class BiGatWithCostAndLambdaActorCritic(_make_advanced_actor_critic(GATConvNet, use_cost_critic=True, use_lambda_net=True)): pass
@ActorCriticRegistry.register('dual_deep_edge_gat_with_cost_and_lambda')
class BiDeepEdgeFeatureGatWithCostAndLambdaActorCritic(_make_advanced_actor_critic(DeepEdgeFeatureGAT, use_cost_critic=True, use_lambda_net=True)): pass


class Actor(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, num_layers=3, gnn_class=GCNConvNet, v_gnn_class=None, p_gnn_class=None, **kwargs):
        super().__init__()
        # Remove gnn_class from kwargs to avoid duplicate propagation
        kwargs = {k: v for k, v in kwargs.items() if k != 'gnn_class'}
        v_gnn_class = v_gnn_class or gnn_class
        p_gnn_class = p_gnn_class or gnn_class
        self.encoder = BiGnnBaseModel(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, num_layers=num_layers, gnn_class=gnn_class, v_gnn_class=v_gnn_class, p_gnn_class=p_gnn_class, **kwargs)

    def forward(self, obs):
        return self.encoder(obs)


class Critic(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, num_layers=3, gnn_class=GCNConvNet, v_gnn_class=None, p_gnn_class=None, **kwargs):
        super().__init__()
        # Remove gnn_class from kwargs to avoid duplicate propagation
        kwargs = {k: v for k, v in kwargs.items() if k != 'gnn_class'}
        v_gnn_class = v_gnn_class or gnn_class
        p_gnn_class = p_gnn_class or gnn_class
        self.encoder = BiGnnBaseModel(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, num_layers=num_layers, gnn_class=gnn_class, v_gnn_class=v_gnn_class, p_gnn_class=p_gnn_class, **kwargs)
                                      
    def forward(self, obs):
        fusion = self.encoder(obs)
        return torch.mean(fusion, dim=-1, keepdim=True)
        

class BiGnnBaseModel(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, num_layers=3, gnn_class=GCNConvNet, v_gnn_class=None, p_gnn_class=None, **kwargs):
        super().__init__()
        v_gnn_class = v_gnn_class or gnn_class
        p_gnn_class = p_gnn_class or gnn_class
        self.v_net_encoder = NetEncoder(v_net_x_dim, v_net_edge_dim, embedding_dim, dropout_prob, batch_norm, num_layers, v_gnn_class)
        self.p_net_encoder = NetEncoder(p_net_x_dim, p_net_edge_dim, embedding_dim, dropout_prob, batch_norm, num_layers, p_gnn_class)
        self.lin = nn.Sequential(nn.Linear(
            embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 1))
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                stdv = 1. / (param.size(-1) ** 0.5)
                param.data.uniform_(-stdv, stdv)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, obs):
        v_net, p_net, curr_v_node_id = obs['v_net'], obs['p_net'], obs['curr_v_node_id']
        v_node_emb, v_g_emb, v_node_dense, _ = self.v_net_encoder(v_net)
        p_node_emb, p_g_emb, p_node_dense, _ = self.p_net_encoder(p_net)
        curr_v_node_id = curr_v_node_id.unsqueeze(1).unsqueeze(1).long()
        curr_v_node_emb = v_node_dense.gather(1, curr_v_node_id.expand(
            v_node_dense.size(0), -1, v_node_dense.size(-1))).squeeze(1)
        p_node_dense = p_node_dense + \
            v_g_emb.unsqueeze(1) + curr_v_node_emb.unsqueeze(1)
        return self.lin(p_node_dense).squeeze(-1)



class NetEncoder(nn.Module):
    def __init__(self, feat_dim, edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, num_layers=3, gnn_class=GCNConvNet):
        super().__init__()
        self.init_lin = nn.Linear(feat_dim, embedding_dim)
        self.net_gnn = gnn_class(embedding_dim, embedding_dim, num_layers=num_layers,
                                 embedding_dim=embedding_dim, edge_dim=edge_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.mean_pool = GraphPooling('mean')
        self.att_pool = GraphPooling('att', output_dim=embedding_dim)

    def forward(self, net_batch):
        x = self.init_lin(net_batch.x)
        net_batch = net_batch.clone()
        net_batch.x = x
        node_emb = self.net_gnn(net_batch)
        g_emb = self.mean_pool(node_emb, net_batch.batch) + \
            self.att_pool(node_emb, net_batch.batch)
        node_dense, _ = to_dense_batch(node_emb, net_batch.batch)
        node_init_dense, _ = to_dense_batch(x, net_batch.batch)
        return node_emb, g_emb, node_dense + g_emb.unsqueeze(1) + node_init_dense, node_init_dense
