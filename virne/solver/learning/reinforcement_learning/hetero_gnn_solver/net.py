import math
from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, to_hetero, HeteroConv
from torch_geometric.utils import to_dense_batch

from .edge_gnn import EdgeGNNConv
from ...neural_network import *
from ...rl_core.base_policy import BaseActorCritic


class ActorCritic(BaseActorCritic):
    
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(ActorCritic, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder = StateEncoder(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.actor = Actor(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic = Critic(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.cost_critic = Critic(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.lambda_net = Critic(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)

    def act(self, x):
        x = self.encoder(x)
        return self.actor(x)
    
    def evaluate(self, x):
        x = self.encoder(x)
        return self.critic(x)

    def evaluate_cost(self, x):
        x = self.encoder(x)
        return self.cost_critic(x)

    def evaluate_lambda(self, x):
        x = self.encoder(x)
        return self.lambda_net(x)

    def contrastive_learning(self, x):
        obs_a = {'hetero_data': x['aug_hetero_data_a']}
        obs_b = {'hetero_data': x['aug_hetero_data_b']}
        p_node_dense_embeddings_a = self.encoder(obs_a).reshape(-1, self.embedding_dim)
        p_node_dense_embeddings_b = self.encoder(obs_b).reshape(-1, self.embedding_dim)
        return p_node_dense_embeddings_a, p_node_dense_embeddings_b


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(Actor, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, state_embeddings):
        logits = self.head(state_embeddings).squeeze(-1)
        return logits


class Critic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(Critic, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1)
        )
        # self.value_head = nn.Sequential(
        #     nn.Linear(p_net_num_nodes, p_net_num_nodes),
        #     nn.LeakyReLU(),
        #     nn.Linear(p_net_num_nodes, 1)
        # )

    def forward(self, state_embeddings):
        fusion = self.head(state_embeddings).squeeze(-1)
        # value = self.value_head(fusion).squeeze(-1)
        value = torch.mean(fusion, dim=-1, keepdim=True)
        return value


class HeteroGnnEncoder(nn.Module):

    gnn_conv = GATConv
    # gnn_conv = EdgeGNNConv

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(HeteroGnnEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(3):
            conv = HeteroConv({
                ('v', 'imaginary', 'p'): self.gnn_conv((-1, -1), embedding_dim, edge_dim=1, add_self_loops=False),
                ('p', 'imaginary', 'v'): self.gnn_conv((-1, -1), embedding_dim, edge_dim=1, add_self_loops=False),
                ('v', 'mapping', 'p'): self.gnn_conv((-1, -1), embedding_dim, edge_dim=1, add_self_loops=False),
                ('p', 'mapping', 'v'): self.gnn_conv((-1, -1), embedding_dim, edge_dim=1, add_self_loops=False),
                ('v', 'connect', 'v'): self.gnn_conv(-1, embedding_dim, edge_dim=1),
                ('p', 'connect', 'p'): self.gnn_conv(-1, embedding_dim, edge_dim=1),
            }, aggr='sum')
            self.convs.append(conv)
        self.init_v_lin = nn.Linear(v_net_x_dim, embedding_dim)
        self.init_p_lin = nn.Linear(p_net_x_dim, embedding_dim)
        self.out_p_lin = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, hetero_data):
        x_dict = hetero_data.x_dict
        v_init_node_embeddings = self.init_v_lin(x_dict['v'])
        p_init_node_embeddings = x_dict['p'] = self.init_p_lin(x_dict['p'])
        x_dict = {'v': v_init_node_embeddings, 'p': p_init_node_embeddings}
        edge_index_dict = hetero_data.edge_index_dict
        edge_attr_dict = hetero_data.edge_attr_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        p_batch = hetero_data.batch_dict['p']
        p_node_embeddings = self.out_p_lin(x_dict['p']) + p_init_node_embeddings
        p_dense_node_embeddings, _ = to_dense_batch(p_node_embeddings, p_batch)
        return p_dense_node_embeddings


class StateEncoder(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, **kwargs):
        super(StateEncoder, self).__init__()
        self.hetero_gnn_encoder = HeteroGnnEncoder(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False)

    def forward(self, obs):
        hetero_data = obs['hetero_data']
        state_embeddings = self.hetero_gnn_encoder(hetero_data)
        return state_embeddings
