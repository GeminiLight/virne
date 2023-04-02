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

from base.environment import SolutionStepEnvironment
from solver import registry
from base import Solution
from solver.learning.buffer import RolloutBuffer
from .sub_env import SubEnv
from .net import Actor
from ..rl_solver import InstanceAgent, RLSolver, PGSolver


@registry.register(
    solver_name='pg_seq2seq',
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning')
class PGSeq2SeqSolver(InstanceAgent, PGSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Sequence-to-Sequence (Seq2Seq) as the neural network model.
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self)
        PGSolver.__init__(self, controller, recorder, counter, **kwargs)
        feature_dim = 3  # (n_attrs, e_attrs, dist, degree)
        action_dim = 100
        self.policy = Actor(feature_dim, action_dim, self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_actor},
        ])
        self.SubEnv = SubEnv
        self.preprocess_obs = obs_as_tensor
        self.compute_advantage_method = 'mc'

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.SubEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        sub_obs = sub_env.get_observation()
        sub_done = False
        outputs = self.policy.encode(self.preprocess_obs(sub_obs, device=self.device))
        p_node_id = p_net.num_nodes
        while not sub_done:
            hidden_state, cell_state = self.policy.get_last_rnn_state()
            sub_obs = {
                'p_node_id': p_node_id,
                'hidden_state': np.squeeze(hidden_state.cpu().detach().numpy(), axis=0),
                'cell_state': np.squeeze(cell_state.cpu().detach().numpy(), axis=0),
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



    def learn_with_instance(self, instance, revenue2cost_list, epoch_logprobs):
        ### -- baseline -- ##
        if self.use_baseline_solver: # or self.use_negative_sample
            baseline_solution = self.baseline_solver.solve(instance)
            baseline_solution_info = self.counter.count_solution(instance['v_net'], baseline_solution)
        else:
            baseline_solution_info = {
                'result': True,
                'v_net_r2c_ratio': 0
            }
        # sub env for sub agent
        sub_buffer = RolloutBuffer()
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.SubEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        sub_obs = sub_env.get_observation()
        sub_done = False
        outputs = self.policy.encode(self.preprocess_obs(sub_obs, device=self.device))
        p_node_id = p_net.num_nodes
        while not sub_done:
            hidden_state, cell_state = self.policy.get_last_rnn_state()
            sub_obs = {
                'p_node_id': p_node_id,
                'hidden_state': np.squeeze(hidden_state.cpu().detach().numpy(), axis=0),
                'cell_state': np.squeeze(cell_state.cpu().detach().numpy(), axis=0),
            }
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            tensor_sub_obs = self.preprocess_obs(sub_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_sub_obs, mask=mask, sample=True)
            value = self.estimate_obs(tensor_sub_obs) if hasattr(self.policy, 'evaluate') else None
            next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action[0])

            p_node_id = action[0].item()

            sub_buffer.add(sub_obs, action, sub_reward, sub_done, action_logprob, value=value)
            sub_buffer.action_masks.append(mask)

            if sub_done:
                break

            sub_obs = next_sub_obs

        solution = sub_env.solution
        if self.use_negative_sample:
            if baseline_solution_info['result'] or sub_env.solution['result']:
                revenue2cost_list.append(sub_reward)
                last_value = self.estimate_obs(self.preprocess_obs(next_sub_obs, self.device)) if hasattr(self.policy, 'evaluate') else None
                sub_buffer.compute_returns_and_advantages(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method)
                self.buffer.merge(sub_buffer)
                epoch_logprobs += sub_buffer.logprobs
                self.time_step += 1
            else:
                pass
        elif sub_env.solution['result']:  #  or True
            revenue2cost_list.append(sub_reward)
            sub_buffer.compute_mc_returns(gamma=self.gamma)
            self.buffer.merge(sub_buffer)
            epoch_logprobs += sub_buffer.logprobs
            self.time_step += 1
        else:
            pass
        return solution

def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        tensor_obs = {}
        for key in obs:
            tensor_obs[key] = obs[key]
            if key == 'p_node_id':
                tensor_obs[key] = torch.LongTensor([obs[key]]).to(device)
            elif key == 'obs':
                tensor_obs[key] = torch.FloatTensor(obs[key]).unsqueeze(dim=0).to(device)
            elif key == 'hidden_state' or key == 'cell_state':
                tensor_obs[key] = torch.FloatTensor(obs[key]).unsqueeze(dim=1).to(device)
        return tensor_obs
    # batch
    elif isinstance(obs, list):
        obs_batch = obs
        tensor_obs = {}
        for key in obs_batch[0]:
            tensor_obs[key] = [one_obs[key] for one_obs in obs_batch]
        for key in tensor_obs:
            if key == 'p_node_id':
                tensor_obs[key] = torch.LongTensor(np.array(tensor_obs[key])).to(device)
            elif key == 'obs':
                tensor_obs[key] = torch.FloatTensor(np.array(tensor_obs[key])).to(device)
            elif key == 'hidden_state' or key == 'cell_state':
                tensor_obs[key] = torch.FloatTensor(np.array(tensor_obs[key])).permute(1, 0, 2).to(device)
        return tensor_obs
    else:
        raise ValueError('obs type error')