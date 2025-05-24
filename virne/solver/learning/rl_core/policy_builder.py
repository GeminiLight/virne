from typing import Optional, Tuple, Any, Dict
import torch
import torch.nn as nn

from virne.solver.learning.rl_policy.gnn_mlp_policy import DeepEdgeFeatureGATActorCritic
from ..rl_policy import GcnMlpActorCritic, GatMlpActorCritic, MlpActorCritic, CnnActorCritic, AttActorCritic
from ..rl_policy import BiGcnActorCritic, BiGatActorCritic, BiDeepEdgeFeatureGatActorCritic
from ..utils import get_pyg_data
from ..obs_handler import POSITIONAL_EMBEDDING_DIM, P_NODE_STATUS_DIM, V_NODE_STATUS_DIM, V_NET_STATUS_DIM
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor


class Encoder(nn.Module):

    def __init__(self, v_net_x_dim, embedding_dim=128):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(v_net_x_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        embeddings = torch.nn.functional.relu(self.emb(x))
        outputs, hidden_state = self.gru(embeddings)
        return outputs, hidden_state



class Seq2SeqActorCriticWrapper(nn.Module):
    def __init__(self, actor_critic_cls, **kwargs):
        actor_critic_cls.__init__(**kwargs)
        v_net_x_dim = kwargs.get('v_net_x_dim', None)
        embedding_dim = kwargs.get('embedding_dim', 128)
        self.encoder = Encoder(v_net_x_dim, embedding_dim=embedding_dim)
        self._last_hidden_state = None

    def encode(self, obs):
        x = obs['v_net_x']
        outputs, hidden_state = self.encoder(x)
        self._last_hidden_state = hidden_state
        return outputs

    def get_last_rnn_state(self):
        return self._last_hidden_state

    def set_last_rnn_hidden(self, hidden_state):
        self._last_hidden_state = hidden_state



class PolicyBuilder:

    @staticmethod
    def get_feature_dim_config(config: Any) -> Dict[str, int]:
        """
        Get the feature dimensions for different components of the policy.
        """
        return {
            'p_net_x_dim': get_p_net_x_dim(config),
            'p_net_edge_dim': get_p_net_edge_dim(config),
            'v_net_x_dim': get_v_net_x_dim(config),
            'v_net_edge_dim': get_v_net_edge_dim(config),
            'v_node_x_dim': get_v_node_x_dim(config),
            'p_net_num_nodes': config.simulation.p_net_setting_num_nodes,
        }

    @staticmethod
    def get_general_nn_config(config: Any) -> Dict[str, Any]:
        """
        Get the general neural network configuration.
        """
        return {
            'embedding_dim': config.nn.embedding_dim,
            'dropout_prob': config.nn.dropout_prob,
            'batch_norm': config.nn.batch_norm
        }

    @staticmethod
    def build_mlp_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build an MLP-based actor-critic policy and its optimizer.
        """
        p_net_x_dim = get_p_net_x_dim(agent.config)
        v_node_feature_dim = get_v_node_x_dim(agent.config)
        policy = MlpActorCritic(
            feature_dim=p_net_x_dim + v_node_feature_dim,
            action_dim=agent.config.simulation.p_net_setting_num_nodes,
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)
        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer

    @staticmethod
    def build_cnn_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build a CNN-based actor-critic policy and its optimizer.
        """
        p_net_x_dim = get_p_net_x_dim(agent.config)
        v_node_feature_dim = get_v_node_x_dim(agent.config)
        policy = CnnActorCritic(
            feature_dim=p_net_x_dim + v_node_feature_dim,
            action_dim=agent.config.simulation.p_net_setting_num_nodes,
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)
        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer
    
    @staticmethod
    def build_gcn_seq2seq_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build an Seq2seq-based actor-critic policy and its optimizer.
        """
        p_net_x_dim = get_p_net_x_dim(agent.config)
        v_net_x_dim = get_v_net_x_dim(agent.config)
        v_node_feature_dim = get_v_node_x_dim(agent.config)
        p_num_nodes = agent.config.simulation.p_net_setting_num_nodes,
        policy = Seq2SeqActorCriticWrapper(
            # actor_critic_cls=agent.actor_critic_cls,
            GcnMlpActorCritic,
            p_num_nodes=p_num_nodes,
            p_net_x_dim=p_net_x_dim,
            v_net_x_dim=v_net_x_dim,
            # v_node_feature_dim=v_node_feature_dim,
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)
        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer

    @staticmethod
    def build_gcn_mlp_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build a GCN+MLP-based actor-critic policy and its optimizer.
        """
        p_net_x_dim = get_p_net_x_dim(agent.config)
        v_node_feature_dim = get_v_node_x_dim(agent.config)
        policy = GcnMlpActorCritic(
            p_net_num_nodes=agent.config.simulation.p_net_setting_num_nodes,
            p_net_x_dim=p_net_x_dim,
            v_node_feature_dim=v_node_feature_dim,
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)
        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer

    @staticmethod
    def build_gat_mlp_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build a GAT+MLP-based actor-critic policy and its optimizer.
        """
        p_net_x_dim = get_p_net_x_dim(agent.config)
        v_node_feature_dim = get_v_node_x_dim(agent.config)
        policy = GatMlpActorCritic(
            p_net_num_nodes=agent.config.simulation.p_net_setting_num_nodes,
            p_net_x_dim=p_net_x_dim,
            v_node_feature_dim=v_node_feature_dim,
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)

        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer


    @staticmethod
    def build_deep_edge_gat_mlp_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build an Edge-GCN+MLP-based actor-critic policy and its optimizer.
        """
        p_net_x_dim = get_p_net_x_dim(agent.config)
        p_net_edge_dim = get_p_net_edge_dim(agent.config)
        v_node_feature_dim = get_v_node_x_dim(agent.config)
        policy = DeepEdgeFeatureGATActorCritic(
            p_net_num_nodes=agent.config.simulation.p_net_setting_num_nodes,
            p_net_x_dim=p_net_x_dim,
            v_node_feature_dim=v_node_feature_dim,
            p_net_edge_dim=p_net_edge_dim,
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)

        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer

    @staticmethod
    def build_dual_gcn_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build a Bi-GCN-based actor-critic policy and its optimizer.
        """
        # feature_dim_config = PolicyBuilder.get_feature_dim_config(agent.config)
        policy = BiGcnActorCritic(
            **PolicyBuilder.get_feature_dim_config(agent.config),
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)

        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer

    @staticmethod
    def build_dual_gat_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build a Bi-GCN-based actor-critic policy and its optimizer.
        """
        policy = BiGatActorCritic(
            **PolicyBuilder.get_feature_dim_config(agent.config),
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)
        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer

    @staticmethod
    def build_dual_deep_edge_gat_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build a Bi-GCN-based actor-critic policy and its optimizer.
        """
        policy = BiDeepEdgeFeatureGatActorCritic(
            **PolicyBuilder.get_feature_dim_config(agent.config),
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)
        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer

    @staticmethod
    def build_att_policy(agent: Any) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Build an Attention-based actor-critic policy and its optimizer.
        """
        p_net_x_dim = get_p_net_x_dim(agent.config)
        v_node_feature_dim = get_v_node_x_dim(agent.config)
        policy = AttActorCritic(
            p_net_num_nodes=agent.config.simulation.p_net_setting_num_nodes,
            p_net_x_dim=p_net_x_dim,
            v_node_feature_dim=v_node_feature_dim, 
            **PolicyBuilder.get_general_nn_config(agent.config),
        ).to(agent.device)
        optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
        return policy, optimizer


class OptimizerBuilder:
    @staticmethod
    def build_optimizer(config: Any, policy: nn.Module) -> torch.optim.Optimizer:
        """
        Build an optimizer for the actor and/or critic parameters of the policy.
        """
        # config.learning_rate {actor: 0.001, critic: 0.001}
        
        learning_rate_dict = config.rl.learning_rate
        assert 'actor' in learning_rate_dict and 'critic' in learning_rate_dict, \
            "Learning rate dictionary must contain 'actor' and 'critic' keys."
        param_list = []
        for name, module in policy.named_children():
            lr_module_key = f'{name}'
            if lr_module_key not in learning_rate_dict:
                raise ValueError(f"Learning rate for '{name}' module not found in learning_rate_dict.\n" + \
                                 f"You may implement new sub-modules in policy, please specify their learning rate of {name} in config.")
            else:
                lr_module = learning_rate_dict[lr_module_key]
            param_list.append({'params': module.parameters(), 'lr': lr_module})
        if not param_list:
            raise ValueError("No parameter to optimize: both actor and critic are False.")
        optimizer = torch.optim.Adam(param_list, weight_decay=config.rl.weight_decay)
        return optimizer


def get_p_net_x_dim(config: Any) -> int:
    """
    Calculate the physical network node feature dimension based on agent config.
    """
    num_p_node_attrs = config.rl.feature_constructor.num_extracted_p_node_attrs
    num_p_link_attrs = config.rl.feature_constructor.num_extracted_p_link_attrs

    if_use_node_status_flags = config.rl.feature_constructor.if_use_node_status_flags
    if_use_aggregated_link_attrs = config.rl.feature_constructor.if_use_aggregated_link_attrs
    if_use_degree_metric = config.rl.feature_constructor.if_use_degree_metric
    if_use_more_topological_metrics = config.rl.feature_constructor.if_use_more_topological_metrics

    p_net_x_dim = num_p_node_attrs + 1  # avg distance
    p_net_x_dim += 2 if if_use_node_status_flags else 0
    p_net_x_dim += num_p_link_attrs * 4 if if_use_aggregated_link_attrs else 0
    p_net_x_dim += 1 if if_use_degree_metric else 0
    p_net_x_dim += 3 if if_use_more_topological_metrics else 0
    return p_net_x_dim

def get_p_net_edge_dim(config: Any) -> int:
    """
    Calculate the physical network edge feature dimension based on agent config.
    """
    num_p_link_attrs = config.rl.feature_constructor.num_extracted_p_link_attrs
    p_net_edge_dim = num_p_link_attrs
    return p_net_edge_dim

def get_v_node_x_dim(config: Any) -> int:
    """
    Calculate the virtual node feature dimension based on agent config.
    """
    num_v_node_attrs = config.rl.feature_constructor.num_extracted_v_node_attrs
    num_v_link_attrs = config.rl.feature_constructor.num_extracted_v_link_attrs

    if_use_node_status_flags = config.rl.feature_constructor.if_use_node_status_flags
    if_use_aggregated_link_attrs = config.rl.feature_constructor.if_use_aggregated_link_attrs
    if_use_degree_metric = config.rl.feature_constructor.if_use_degree_metric
    if_use_more_topological_metrics = config.rl.feature_constructor.if_use_more_topological_metrics

    v_node_x_dim = num_v_node_attrs + 1  # num_neighbors
    v_node_x_dim += 3 if if_use_node_status_flags else 0
    v_node_x_dim += num_v_link_attrs * 4 if if_use_aggregated_link_attrs else 0
    v_node_x_dim += 0 if if_use_degree_metric else 0
    v_node_x_dim += 0 if if_use_more_topological_metrics else 0
    return v_node_x_dim

def get_v_net_x_dim(config: Any) -> int:
    """
    Calculate the virtual node feature dimension based on agent config.
    """
    num_v_node_attrs = config.rl.feature_constructor.num_extracted_v_node_attrs
    num_v_link_attrs = config.rl.feature_constructor.num_extracted_v_link_attrs

    if_use_node_status_flags = config.rl.feature_constructor.if_use_node_status_flags
    if_use_aggregated_link_attrs = config.rl.feature_constructor.if_use_aggregated_link_attrs
    if_use_degree_metric = config.rl.feature_constructor.if_use_degree_metric
    if_use_more_topological_metrics = config.rl.feature_constructor.if_use_more_topological_metrics

    v_node_x_dim = num_v_node_attrs + 1  # neighbor flag
    v_node_x_dim += 3 if if_use_node_status_flags else 0
    v_node_x_dim += num_v_link_attrs * 4 if if_use_aggregated_link_attrs else 0
    v_node_x_dim += 1 if if_use_degree_metric else 0
    v_node_x_dim += 3 if if_use_more_topological_metrics else 0
    return v_node_x_dim

def get_v_net_edge_dim(config: Any) -> int:
    """
    Calculate the virtual network edge feature dimension based on agent config.
    """
    num_v_link_attrs = config.rl.feature_constructor.num_extracted_v_link_attrs
    v_net_edge_dim = num_v_link_attrs
    return v_net_edge_dim
