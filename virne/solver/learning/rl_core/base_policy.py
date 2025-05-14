import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any

from virne.solver.learning.neural_network import *

from typing import Any

class BaseActorCritic(nn.Module):
    def act(self, obs: Any):
        raise NotImplementedError

    def evaluate(self, obs: Any):
        raise NotImplementedError

class ActorCriticWithSharedEncoderBase(nn.Module):

    def __init__(self, ):
        super(ActorCriticWithSharedEncoderBase, self).__init__()
    
    def act(self, x):
        x = self.encoder(x)
        return self.actor(x)
    
    def evaluate(self, x):
        x = self.encoder(x)
        if not hasattr(self, 'critic'):
            return None
        return self.critic(x)


class ActorCriticRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(actor_critic_cls):
            cls._registry[name] = actor_critic_cls
            return actor_critic_cls
        return decorator

    @classmethod
    def get(cls, name):
        if name not in cls._registry:
            raise NotImplementedError(f"ActorCritic '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def list_registered(cls):
        return list(cls._registry.keys())


@ActorCriticRegistry.register('cnn')
class CnnActorCritic(BaseActorCritic):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=128):
        super(CnnActorCritic, self).__init__()
        self.actor = CnnActor(feature_dim, action_dim, embedding_dim)
        self.critic = CnnCritic(feature_dim, action_dim, embedding_dim)


class CnnActor(nn.Module):
    def __init__(self, feature_dim, action_dim, embedding_dim=128):
        super(CnnActor, self).__init__()
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


class CnnCritic(nn.Module):
    def __init__(self, feature_dim, action_dim, embedding_dim=128):
        super(CnnCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[1, feature_dim], stride=[1, 1]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(action_dim, 1)
        )
        
    def forward(self, obs):
        """Return logits of actions"""
        x = obs['p_net_x']
        values = self.net(x)
        return values


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch

from ..neural_network import *


class GcnActorCritic(nn.Module):
    """
    Actor-Critic model using GCN for the substrate (p_net) and MLP for the virtual node (v_net) features.
    Modular, extensible, and clear for RL research and production.
    """
    def __init__(self, p_net_num_nodes: int, p_net_x_dim: int, v_node_feature_dim: int, embedding_dim: int = 128, dropout_prob: float = 0., batch_norm: bool = False, **kwargs):
        super().__init__()
        self.actor = GcnActor(p_net_num_nodes, p_net_x_dim, v_node_feature_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic = GcnCritic(p_net_num_nodes, p_net_x_dim, v_node_feature_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)

    def act(self, obs: dict) -> torch.Tensor:
        """Compute action logits from observation."""
        return self.actor(obs)

    def evaluate(self, obs: dict) -> torch.Tensor:
        """Compute value estimate from observation."""
        return self.critic(obs)

class GcnActor(nn.Module):
    """
    GCN-based actor for node selection. Fuses p_net and v_net features.
    Extensible for custom fusion or additional context.
    """
    def __init__(self, p_net_num_nodes: int, p_net_x_dim: int, v_node_feature_dim: int, embedding_dim: int = 128, dropout_prob: float = 0., batch_norm: bool = False, **kwargs):
        super().__init__()
        self.gnn = GCNConvNet(p_net_x_dim, embedding_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, return_batch=True)
        self.mlp = MLPNet(v_node_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.lin_fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        """Return logits of actions. Expects obs['p_net'] and obs['v_net_x'] as tensors."""
        p_node_embeddings = self.gnn(obs['p_net'])  # (batch, N, D)
        v_node_embedding = self.mlp(obs['v_net_x'])  # (batch, D)
        if p_node_embeddings.shape[0] != v_node_embedding.shape[0]:
            raise ValueError(f"Batch size mismatch: p_node_embeddings {p_node_embeddings.shape}, v_node_embedding {v_node_embedding.shape}")
        fusion_embeddings = p_node_embeddings + v_node_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1)
        action_logits = self.lin_fusion(fusion_embeddings).squeeze(-1)  # (batch, N)
        return action_logits

class GcnCritic(nn.Module):
    """
    GCN-based critic for value estimation. Fuses p_net and v_net features.
    Extensible for custom value heads or aggregation.
    """
    def __init__(self, p_net_num_nodes: int, p_net_x_dim: int, v_node_feature_dim: int, embedding_dim: int = 128, dropout_prob: float = 0., batch_norm: bool = False, **kwargs):
        super().__init__()
        self.gnn = GCNConvNet(p_net_x_dim, embedding_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, return_batch=True)
        self.mlp = MLPNet(v_node_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.lin_fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        """Return value estimate. Expects obs['p_net'] and obs['v_net_x'] as tensors."""
        p_node_embeddings = self.gnn(obs['p_net'])  # (batch, N, D)
        v_node_embedding = self.mlp(obs['v_net_x'])  # (batch, D)
        if p_node_embeddings.shape[0] != v_node_embedding.shape[0]:
            raise ValueError(f"Batch size mismatch: p_node_embeddings {p_node_embeddings.shape}, v_node_embedding {v_node_embedding.shape}")
        fusion_embedding = p_node_embeddings + v_node_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1)
        action_logits = self.lin_fusion(fusion_embedding).squeeze(-1)  # (batch, N)
        value = action_logits.mean(dim=1)  # (batch,)
        return value

