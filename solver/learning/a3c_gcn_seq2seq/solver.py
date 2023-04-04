# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

from base import Solution, SolutionStepEnvironment
from solver import registry
from .sub_env import SubEnv
from .net import ActorCritic, Actor, Critic
from ..buffer import RolloutBuffer
from ..rl_solver import RLSolver, PPOSolver, InstanceAgent, A2CSolver
from ..utils import get_pyg_data


@registry.register(
    solver_name='a3c_gcn_seq2seq', 
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning')
class A3CGcnSeq2SeqSolver(InstanceAgent, A2CSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Advantage Actor-Critic (A3C) as the training algorithm,
    and Graph Convolutional Network (GCN) and Sequence-to-Sequence (Seq2Seq)
    as the neural network model.

    References:
        - Tianfu Wang, et al. "DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning". In ICC, 2021.
        
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self)
        A2CSolver.__init__(self, controller, recorder, counter, **kwargs)
        num_p_net_nodes = kwargs['p_net_setting']['num_nodes']
        self.policy = ActorCritic(p_net_num_nodes=num_p_net_nodes, p_net_feature_dim=5, v_net_feature_dim=2, embedding_dim=self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic},
            ],
        )
        self.SubEnv = SubEnv
        self.preprocess_obs = obs_as_tensor
        self.preprocess_encoder_obs = encoder_obs_to_tensor
        self.compute_advantage_method = 'mc'

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.SubEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        encoder_obs = sub_env.get_observation()
        sub_done = False
        encoder_outputs = self.policy.encode(self.preprocess_encoder_obs(encoder_obs, device=self.device))
        encoder_outputs = encoder_outputs.squeeze(1).cpu().detach().numpy()
        p_node_id = p_net.num_nodes
        while not sub_done:
            sub_obs = encoder_obs
            hidden_state = self.policy.get_last_rnn_state()
            sub_obs = {
                'p_node_id': p_node_id,
                'hidden_state': np.squeeze(hidden_state.cpu().detach().numpy(), axis=0),
                'p_net_x': sub_obs['p_net_x'],
                'p_net_edge_index': sub_obs['p_net_edge_index'],
                'encoder_outputs': encoder_outputs,
            }
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            tensor_sub_obs = self.preprocess_obs(sub_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_sub_obs, mask=mask, sample=True)
            next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action[0])

            p_node_id = action[0].item()

            if sub_done:
                break

            sub_obs = next_sub_obs
        return sub_env.solution

    def learn_with_instance(self, instance):
        # sub env for sub agent
        sub_buffer = RolloutBuffer()
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.SubEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        encoder_obs = sub_env.get_observation()
        sub_done = False
        encoder_outputs = self.policy.encode(self.preprocess_encoder_obs(encoder_obs, device=self.device))
        encoder_outputs = encoder_outputs.squeeze(1).cpu().detach().numpy()
        p_node_id = p_net.num_nodes
        hidden_state = self.policy.get_last_rnn_state()
        sub_obs = {
            'p_node_id': p_node_id,
            'hidden_state': hidden_state.squeeze(0).cpu().detach().numpy(),
            'p_net_x': encoder_obs['p_net_x'],
            'p_net_edge_index': encoder_obs['p_net_edge_index'],
            'encoder_outputs': encoder_outputs,
        }
        while not sub_done:
            hidden_state = self.policy.get_last_rnn_state()
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            tensor_sub_obs = self.preprocess_obs(sub_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_sub_obs, mask=mask, sample=True)
            value = self.estimate_obs(tensor_sub_obs) if hasattr(self.policy, 'evaluate') else None
            next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action[0])

            p_node_id = action[0].item()

            sub_buffer.add(sub_obs, action, sub_reward, sub_done, action_logprob, value=value)
            sub_buffer.action_masks.append(mask)

            next_sub_obs = {
                'p_node_id': p_node_id,
                'hidden_state': hidden_state.squeeze(0).cpu().detach().numpy(),
                'p_net_x': next_sub_obs['p_net_x'],
                'p_net_edge_index': next_sub_obs['p_net_edge_index'],
                'encoder_outputs': encoder_outputs,
            }
            if sub_done:
                break

            sub_obs = next_sub_obs

        last_value = self.estimate_obs(self.preprocess_obs(next_sub_obs, self.device)) if hasattr(self.policy, 'evaluate') else None
        solution = sub_env.solution
        return solution, sub_buffer, last_value


def encoder_obs_to_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        v_net_x = obs['v_net_x']
        obs_v_net_x = torch.FloatTensor(v_net_x).unsqueeze(dim=0).to(device)
        return {'v_net_x': obs_v_net_x}
    elif isinstance(obs, list):
        obs_batch = obs
        v_net_x_list = []
        for observation in obs:
            v_net_x = obs['v_net_x']
            v_net_x_list.append(v_net_x)
        obs_v_net_x = torch.FloatTensor(np.array(v_net_x_list)).to(device)
        return {'v_net_x': obs_v_net_x}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'])
        obs_p_net = Batch.from_data_list([data]).to(device)
        obs_p_node_id = torch.LongTensor([obs['p_node_id']]).to(device)
        obs_hidden_state = torch.FloatTensor(obs['hidden_state']).unsqueeze(dim=0).to(device)
        obs_encoder_outputs = torch.FloatTensor(obs['encoder_outputs']).unsqueeze(dim=0).to(device)
        return {'p_net': obs_p_net, 'p_node_id': obs_p_node_id, 
                'hidden_state': obs_hidden_state, 'encoder_outputs': obs_encoder_outputs,
                'mask': None}
    # batch
    elif isinstance(obs, list):
        obs_batch = obs
        p_net_data_list, p_node_id_list, hidden_state_list, encoder_outputs_list = [], [], [], []
        for observation in obs_batch:
            p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'])
            p_node_id = observation['p_node_id']
            hidden_state = observation['hidden_state']
            encoder_outputs = observation['encoder_outputs']
            p_net_data_list.append(p_net_data)
            p_node_id_list.append(p_node_id)
            hidden_state_list.append(hidden_state)
            encoder_outputs_list.append(encoder_outputs)
        obs_p_node_id = torch.LongTensor(np.array(p_node_id_list)).to(device)
        obs_hidden_state = torch.FloatTensor(np.array(hidden_state_list)).to(device)
        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
        # Pad sequences with zeros and get the mask of padded elements
        sequences = encoder_outputs_list
        max_length = max([seq.shape[0] for seq in sequences])
        padded_sequences = np.zeros((len(sequences), max_length, sequences[0].shape[1]))
        mask = np.zeros((len(sequences), max_length), dtype=np.bool)
        for i, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            padded_sequences[i, :seq_len, :] = seq
            mask[i, :seq_len] = 1
        obs_encoder_outputs = torch.FloatTensor(np.array(padded_sequences)).to(device)
        obs_mask = torch.FloatTensor(mask).to(device)
        return {'p_net': obs_p_net, 'p_node_id': obs_p_node_id, 
                'hidden_state': obs_hidden_state, 'encoder_outputs': obs_encoder_outputs, 
                'mask': obs_mask}
    else:
        raise ValueError('obs type error')
