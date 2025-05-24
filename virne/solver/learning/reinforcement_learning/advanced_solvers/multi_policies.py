from collections import Counter
import copy
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch

from virne.solver.learning.rl_core.buffer import RolloutBuffer
from .instance_env import InstanceRLEnv, InstanceRLEnvWithNrmRank, InstanceRLEnvWithNeaRank
from .net import ActorCritic
from virne.solver.learning.rl_core import RLSolver, PPOSolver, A2CSolver, InstanceAgent, A3CSolver
from ..utils import get_pyg_data
from ..obs_handler import POSITIONAL_EMBEDDING_DIM


@SolverRegistry.register(solver_name='a3c_gcn', solver_type='r_learning')
class A3cGcnSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, InstanceRLEnv)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)


@SolverRegistry.register(solver_name='a3c_gcn_nrm_rank', solver_type='r_learning')
class A3CGcnNrmRankSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, InstanceRLEnvWithNrmRank)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)


@SolverRegistry.register(solver_name='a3c_gcn_nea_rank', solver_type='r_learning')
class A3CGcnNeaRankSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, InstanceRLEnvWithNeaRank)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)


def make_policy(agent, **kwargs):
    num_vn_attrs = agent.config.simulation.v_sim_setting_num_node_resource_attrs
    num_vl_attrs = agent.config.simulation.v_sim_setting_num_link_resource_attrs
    policy = ActorCritic(p_net_num_nodes=agent.config.simulation.p_net_setting_num_nodes, 
                        p_net_x_dim=num_vn_attrs*2 + num_vl_attrs*2 + 1, 
                        v_node_feature_dim=num_vn_attrs+num_vl_attrs+1,
                        embedding_dim=agent.config.nn.embedding_dim, 
                        dropout_prob=agent.config.nn.dropout_prob, 
                        batch_norm=agent.config.nn.batch_norm).to(agent.device)
    optimizer = torch.optim.Adam([
            {'params': policy.actor.parameters(), 'lr': agent.config.rl.learning_rate.actor},
            {'params': policy.critic.parameters(), 'lr': agent.config.rl.learning_rate.critic},
        ], weight_decay=agent.config.rl.weight_decay
    )
    return policy, optimizer


@SolverRegistry.register(solver_name='a3c_gcn_multi_policies', solver_type='r_learning')
class A3CGcnMultiPoliciesSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, InstanceRLEnv)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        # self.config.rl.maskable_policy = False
        self.meta_policy = copy.deepcopy(self.policy)
        self.meta_optimizer = torch.optim.Adam(self.meta_policy.parameters(), lr=self.lr)
        self.task_policies = {}
        self.task_optimizers = {}
        self.target_steps = 1024
        self.infer_with_single_task_policy_id = kwargs.get('infer_with_single_task_policy_id', 0)
        if self.infer_with_single_task_policy_id != 0:
            print(f'Infer with single task policy id: {self.infer_with_single_task_policy_id}') if self.verbose >= 0 else None

    def solve(self, instance):
        v_net_size = instance['v_net'].num_nodes
        if self.infer_with_single_task_policy_id != 0:
            self.searcher.policy = self.task_policies[self.infer_with_single_task_policy_id]
            return super().solve(instance)
        if v_net_size not in self.task_policies:
            self.searcher.policy = self.meta_policy
        else:
            self.searcher.policy = self.task_policies[v_net_size]
        return super().solve(instance)

    def save_model(self, checkpoint_fname):
        checkpoint_fname = os.path.join(self.model_dir, checkpoint_fname)
        model_list = [(task_id, self.task_policies[task_id], self.task_optimizers[task_id]) for task_id in self.task_policies.keys()]
        task_model_dict = {}
        task_model_dict['meta_policy'] = {'policy': self.meta_policy.state_dict(), 'optimizer': self.meta_optimizer.state_dict()}
        task_model_dict['task_policies'] = {}
        for task_id, policy, optimizer in model_list:
            task_model_dict['task_policies'][task_id] = {'policy': policy.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(task_model_dict, checkpoint_fname)
        self.logger.info(f'Save model to {checkpoint_fname}\n')

    def load_model(self, checkpoint_path):
        print('Attempting to load the pretrained model')
        try:
            checkpoint = torch.load(checkpoint_path)
            self.meta_policy.load_state_dict(checkpoint['meta_policy']['policy'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_policy']['optimizer'])
            for task_id in checkpoint['task_policies'].keys():
                if task_id not in self.task_policies:
                    self.task_policies[task_id] = copy.deepcopy(self.meta_policy)
                    self.task_optimizers[task_id] = torch.optim.Adam(self.task_policies[task_id].parameters(), lr=self.lr)
                    print('New task policy is created for task {}'.format(task_id))
                self.task_policies[task_id].load_state_dict(checkpoint['task_policies'][task_id]['policy'])
                self.task_optimizers[task_id].load_state_dict(checkpoint['task_policies'][task_id]['optimizer'])
            self.logger.info(f'Loaded pretrained model from {checkpoint_path}')
        except Exception as e:
            self.logger.info(f'Load failed from {checkpoint_path}\nInitilized with random parameters')

    def learn_with_instance(self, instance):
        # sub env for sub agent
        v_net, p_net = instance['v_net'], instance['p_net']
        v_net_size = v_net.num_nodes
        task_id = v_net_size
        if task_id not in self.task_policies:
            self._init_task_policy_and_task_optimizer(task_id)
        self.policy = self.task_policies[v_net_size]
        self.optimizer = self.task_optimizers[v_net_size]
        return super().learn_with_instance(instance)

    def update(self):
        self._fine_tuning_update()

    def _init_task_policy_and_task_optimizer(self, task_id):
        self.task_policies[task_id] = copy.deepcopy(self.meta_policy)
        self.task_optimizers[task_id] = torch.optim.Adam(self.task_policies[task_id].parameters(), lr=self.lr)
        print(f'New task policy is created for task {task_id}')

    def _stats_task_dist(self, buffer):
        v_net_size_list = np.array([obs['v_net_size'] for obs in buffer.observations])
        return Counter(v_net_size_list)

    def _split_buffer(self, buffer):
        # Split buffer
        task_buffers = {}
        v_net_size_list = np.array([obs['v_net_size'] for obs in buffer.observations])
        tasks_list = sorted(list(set(v_net_size_list)))
        for task_id in tasks_list:
            task_indices = np.where(v_net_size_list == task_id)[0]
            task_buffer = RolloutBuffer()
            task_buffer.observations = [buffer.observations[i] for i in task_indices]
            task_buffer.actions = np.array(buffer.actions)[task_indices].tolist()
            task_buffer.logprobs = np.array(buffer.logprobs)[task_indices].tolist()
            task_buffer.rewards = np.array(buffer.rewards)[task_indices].tolist()
            task_buffer.returns = np.array(buffer.returns)[task_indices].tolist()
            # task_buffer.action_masks = [buffer.action_masks[i] for i in task_indices]
            task_buffers[task_id] = task_buffer
        task_buffers = {k: task_buffers[k] for k in sorted(list(task_buffers.keys()))}
        return task_buffers

    def _fine_tuning_update(self):
        meta_buffer = self.buffer
        # Initialize task policies
        task_dist = self._stats_task_dist(self.buffer)
        print(f'Task distribution: {task_dist}')
        for task_id in task_dist.keys():
            if task_id not in self.task_policies:
                self.task_policies[task_id] = copy.deepcopy(self.meta_policy)
                self.task_optimizers[task_id] = torch.optim.Adam(self.task_policies[task_id].parameters(), lr=self.lr)
        # Split buffer
        task_buffers = self._split_buffer(self.buffer)
        # Inner loop
        for task_id in task_buffers.keys():
            self.policy = self.task_policies[task_id]
            self.optimizer = self.task_optimizers[task_id]
            self.buffer = task_buffers[task_id]
            super().update()
        meta_buffer.clear()

