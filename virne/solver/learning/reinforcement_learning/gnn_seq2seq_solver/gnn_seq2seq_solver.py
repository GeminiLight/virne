# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================

import numpy as np
from gym import spaces
from typing import Any, Dict, Tuple, List, Union, Optional, Type, Callable


import os
import torch
import numpy as np
from omegaconf import open_dict

from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger

from virne.solver import SolverRegistry
from virne.solver.learning.rl_policy import CnnActorCritic
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.policy_builder import OptimizerBuilder, PolicyBuilder
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry, PNetVNodeFeatureConstructor, PNetVNetFeatureConstructor
from virne.solver.learning.rl_core.reward_calculator import RewardCalculatorRegistry, GradualIntermediateRewardCalculator
from virne.solver.learning.reinforcement_learning.solver_maker import make_solver_class

from virne.solver.base_solver import SolverRegistry
from virne.solver.learning.rl_core import RLSolver, PPOSolver, InstanceAgent, A2CSolver, RolloutBuffer

from .policy_with_encoder import GcnSeq2SeqActorCritic, GATSeq2SeqActorCritic


obs_as_tensor = TensorConvertor.obs_as_tensor_for_gnn_seq2seq
encoder_obs_to_tensor = TensorConvertor.v_net_x_obs_as_tensor

class A3CGcnSeq2SeqInstanceEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'
        super(A3CGcnSeq2SeqInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


@SolverRegistry.register(solver_name='ppo_gcn_seq2seq+', solver_type='r_learning')
class PpoGcnSeq2SeqSolver(InstanceAgent, PPOSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Advantage Actor-Critic (A3C) as the training algorithm,
    and Graph Convolutional Network (GCN) and Sequence-to-Sequence (Seq2Seq)
    as the neural network model.

    References:
        - Tianfu Wang, et al. "DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning". In ICC, 2021.
        
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, A3CGcnSeq2SeqInstanceEnv)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.preprocess_encoder_obs = encoder_obs_to_tensor

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
        encoder_obs = sub_env.get_observation()
        instance_done = False
        encoder_outputs = self.policy.encode(self.preprocess_encoder_obs(encoder_obs, device=self.device))
        encoder_outputs = encoder_outputs.squeeze(1).cpu().detach().numpy()
        p_node_id = p_net.num_nodes
        hidden_state = self.policy.get_last_rnn_state()
        encoder_obs['p_node_id'] = p_node_id
        encoder_obs['hidden_state'] = np.squeeze(hidden_state.cpu().detach().numpy(), axis=0)
        encoder_obs['encoder_outputs'] = encoder_outputs
        encoder_obs['action_mask'] = np.expand_dims(sub_env.generate_action_mask(), axis=0)
        instance_obs = encoder_obs
        while not instance_done:
            hidden_state = self.policy.get_last_rnn_state()
            tensor_instance_obs = self.preprocess_obs(instance_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_instance_obs, sample=True)
            next_instance_obs, instance_reward, instance_done, instance_info = sub_env.step(action)

            next_instance_obs['p_node_id'] = action
            next_instance_obs['hidden_state'] = np.squeeze(hidden_state.cpu().detach().numpy(), axis=0)
            next_instance_obs['encoder_outputs'] = encoder_outputs
            next_instance_obs['action_mask'] = np.expand_dims(sub_env.generate_action_mask(), axis=0)

            if instance_done:
                break

            instance_obs = next_instance_obs
        return sub_env.solution

    def learn_with_instance(self, instance):
        # sub env for sub agent
        sub_buffer = RolloutBuffer()
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
        encoder_obs = sub_env.get_observation()
        instance_done = False
        encoder_outputs = self.policy.encode(self.preprocess_encoder_obs(encoder_obs, device=self.device))
        encoder_outputs = encoder_outputs.squeeze(1).cpu().detach().numpy()
        p_node_id = p_net.num_nodes
        hidden_state = self.policy.get_last_rnn_state()
        encoder_obs['p_node_id'] = p_node_id
        encoder_obs['hidden_state'] = np.squeeze(hidden_state.cpu().detach().numpy(), axis=0)
        encoder_obs['encoder_outputs'] = encoder_outputs
        encoder_obs['action_mask'] = np.expand_dims(sub_env.generate_action_mask(), axis=0)
        instance_obs = encoder_obs
        while True:
            hidden_state = self.policy.get_last_rnn_state()
            tensor_instance_obs = self.preprocess_obs(instance_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_instance_obs, sample=True)
            value = self.estimate_value(tensor_instance_obs) if hasattr(self.policy, 'evaluate') else None
            next_instance_obs, instance_reward, instance_done, instance_info = sub_env.step(action)
        
            sub_buffer.add(instance_obs, action, instance_reward, instance_done, action_logprob, value=value)
            next_instance_obs['p_node_id'] = action
            next_instance_obs['hidden_state'] = np.squeeze(hidden_state.cpu().detach().numpy(), axis=0)
            next_instance_obs['encoder_outputs'] = encoder_outputs
            next_instance_obs['action_mask'] = np.expand_dims(sub_env.generate_action_mask(), axis=0)

            if instance_done:
                break

            instance_obs = next_instance_obs

        last_value = self.estimate_value(self.preprocess_obs(next_instance_obs, self.device)) if hasattr(self.policy, 'evaluate') else None
        solution = sub_env.solution
        return solution, sub_buffer, last_value


def make_policy(agent, **kwargs):
    feature_dim_config = PolicyBuilder.get_feature_dim_config(agent.config)
    general_nn_config = PolicyBuilder.get_general_nn_config(agent.config)
    policy = GcnSeq2SeqActorCritic(
        p_net_num_nodes=feature_dim_config['p_net_num_nodes'], 
        p_net_x_dim=feature_dim_config['p_net_x_dim'],
        p_net_edge_dim=feature_dim_config['p_net_edge_dim'],
        v_net_x_dim=feature_dim_config['v_net_x_dim'], 
        **general_nn_config).to(agent.device)
    optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
    return policy, optimizer


@SolverRegistry.register(solver_name='ppo_gat_seq2seq+', solver_type='r_learning')
class PpoGatSeq2SeqSolver(PpoGcnSeq2SeqSolver):
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        InstanceAgent.__init__(self, A3CGcnSeq2SeqInstanceEnv)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy_gat, obs_as_tensor, **kwargs)
        self.preprocess_encoder_obs = encoder_obs_to_tensor

def make_policy_gat(agent, **kwargs):
    feature_dim_config = PolicyBuilder.get_feature_dim_config(agent.config)
    general_nn_config = PolicyBuilder.get_general_nn_config(agent.config)
    policy = GATSeq2SeqActorCritic(
        p_net_num_nodes=feature_dim_config['p_net_num_nodes'], 
        p_net_x_dim=feature_dim_config['p_net_x_dim'],
        p_net_edge_dim=feature_dim_config['p_net_edge_dim'],
        v_net_x_dim=feature_dim_config['v_net_x_dim'], 
        **general_nn_config).to(agent.device)
    optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
    return policy, optimizer