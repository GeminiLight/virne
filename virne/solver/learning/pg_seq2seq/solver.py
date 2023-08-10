# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from virne.base.environment import SolutionStepEnvironment
from virne.solver import registry
from virne.base import Solution
from virne.solver.learning.rl_base.buffer import RolloutBuffer
from .instance_env import InstanceEnv
from .net import ActorCritic
from virne.solver.learning.rl_base import InstanceAgent, RLSolver, PGSolver


@registry.register(
    solver_name='pg_seq2seq',
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning')
class PgSeq2SeqSolver(InstanceAgent, PGSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Sequence-to-Sequence (Seq2Seq) as the neural network model.
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self, InstanceEnv)
        PGSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        self.preprocess_encoder_obs = encoder_obs_to_tensor
        self.compute_advantage_method = 'mc'

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        instance_obs = sub_env.get_observation()
        instance_done = False
        outputs = self.policy.encode(self.preprocess_encoder_obs(instance_obs, device=self.device))
        p_node_id = p_net.num_nodes
        while not instance_done:
            hidden_state, cell_state = self.policy.get_last_rnn_state()
            instance_obs = {
                'p_node_id': p_node_id,
                'hidden_state': np.squeeze(hidden_state.cpu().detach().numpy(), axis=0),
                'cell_state': np.squeeze(cell_state.cpu().detach().numpy(), axis=0),
            }
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            tensor_instance_obs = self.preprocess_obs(instance_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_instance_obs, mask=mask, sample=True)
            next_instance_obs, instance_reward, instance_done, instance_info = sub_env.step(action)

            p_node_id = action.item()

            if instance_done:
                break

            instance_obs = next_instance_obs
        return sub_env.solution

    def learn_with_instance(self, instance):
        # sub env for sub agent
        sub_buffer = RolloutBuffer()
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        instance_obs = sub_env.get_observation()
        instance_done = False
        outputs = self.policy.encode(self.preprocess_encoder_obs(instance_obs, device=self.device))
        p_node_id = p_net.num_nodes
        while not instance_done:
            hidden_state, cell_state = self.policy.get_last_rnn_state()
            instance_obs = {
                'p_node_id': p_node_id,
                'hidden_state': np.squeeze(hidden_state.cpu().detach().numpy(), axis=0),
                'cell_state': np.squeeze(cell_state.cpu().detach().numpy(), axis=0),
                'action_mask': np.expand_dims(sub_env.generate_action_mask(), axis=0)
            }
            tensor_instance_obs = self.preprocess_obs(instance_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_instance_obs, mask=mask, sample=True)
            value = self.estimate_value(tensor_instance_obs) if hasattr(self.policy, 'evaluate') else None
            next_instance_obs, instance_reward, instance_done, instance_info = sub_env.step(action)

            p_node_id = action.item()

            sub_buffer.add(instance_obs, action, instance_reward, instance_done, action_logprob, value=value)
            sub_buffer.action_masks.append(mask)

            if instance_done:
                break

            instance_obs = next_instance_obs

        last_value = self.estimate_value(self.preprocess_obs(next_instance_obs, self.device)) if hasattr(self.policy, 'evaluate') else None
        solution = sub_env.solution
        return solution, sub_buffer, last_value


def make_policy(agent, **kwargs):
    feature_dim = 3  # (n_attrs, e_attrs, dist, degree)
    action_dim = 100
    policy = ActorCritic(feature_dim, action_dim, agent.embedding_dim).to(agent.device)
    optimizer = torch.optim.Adam([
        {'params': policy.parameters(), 'lr': agent.lr_actor},
    ], weight_decay=agent.weight_decay)
    return policy, optimizer


def encoder_obs_to_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        obs_p_net_x = torch.FloatTensor(obs['p_net_x']).unsqueeze(dim=0).to(device)
        return {'p_net_x': obs_p_net_x}
    # batch
    elif isinstance(obs, list):
        obs_batch = obs
        p_net_x_list = []
        for observation in obs_batch:
            p_net_x = observation['p_net_x']
            p_net_x_list.append(p_net_x)
        obs_p_net_x = torch.FloatTensor(np.array(p_net_x_list)).to(device)
        return {'p_net_x': obs_p_net_x}


def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        obs_p_node_id = torch.LongTensor([obs['p_node_id']]).to(device)
        obs_hidden_state = torch.FloatTensor(obs['hidden_state']).unsqueeze(dim=1).to(device)
        obs_cell_state = torch.FloatTensor(obs['cell_state']).unsqueeze(dim=1).to(device)
        return {'p_node_id': obs_p_node_id, 'hidden_state': obs_hidden_state, 'cell_state': obs_cell_state}
    # batch
    elif isinstance(obs, list):
        obs_batch = obs
        p_node_id_list, hidden_state_list, cell_state_list = [], [], []
        for observation in obs_batch:
            p_node_id_list.append(observation['p_node_id'])
            hidden_state_list.append(observation['hidden_state'])
            cell_state_list.append(observation['cell_state'])
        obs_p_node_id = torch.LongTensor(np.array(p_node_id_list)).to(device)
        obs_hidden_state = torch.FloatTensor(np.array(hidden_state_list)).permute(1, 0, 2).to(device)
        obs_cell_state = torch.FloatTensor(np.array(cell_state_list)).permute(1, 0, 2).to(device)
        return {'p_node_id': obs_p_node_id, 'hidden_state': obs_hidden_state, 'cell_state': obs_cell_state}
    else:
        raise ValueError('obs type error')