# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import csv
import copy
import math
import time
from collections.abc import Mapping
from omegaconf import OmegaConf, open_dict
from sympy import im
import tqdm
import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pool
from torch.distributions import Categorical
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from abc import abstractmethod

from virne.solver import Solver
from virne.solver.heuristic.node_rank import *

from .searcher import *
from .buffer import RolloutBuffer
from .shared_adam import SharedAdam, sync_gradients
from ..utils import apply_mask_to_logit, get_observations_sample, RunningMeanStd
from virne.utils import test_running_time


class RLSolver(Solver):
    """General Reinforcement Learning Solve"""
    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(RLSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        if self.allow_rejection or self.allow_revocable:
            raise NotImplementedError(
                'RL policies currently emit one logit per physical node, so '
                'solver.allow_rejection and solver.allow_revocable are not '
                'supported. Keep both options false.'
            )
        self.rank = 0
        # baseline
        self.if_use_baseline_solver = self.config.rl.if_use_baseline_solver
        self.if_allow_baseline_unsafe_solve = self.config.rl.if_allow_baseline_unsafe_solve
        if self.if_use_baseline_solver:
            self.baseline_solvers = {}
            baselin_solver_name = self.config.rl.baselin_solver_name
            if baselin_solver_name == 'grc':
                self.baseline_solver = GRCRankSolver(controller, recorder, counter, logger, config, **kwargs)
            elif baselin_solver_name == 'self':
                self.baseline_solver = copy.deepcopy(self)
        # training
        self.use_cuda = self.config.training.use_cuda
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.config.training.gpu_id}')
            self.device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            self.device = torch.device('cpu')
            self.device_name = 'CPU'
            self.use_cuda = False
        self.num_workers = kwargs.get('num_workers', 1)
        if self.num_workers == 1:
            with open_dict(self.config):
                self.config.training.distributed_training = False
        # rl
        self.gae_lambda = self.config.rl.gae_lambda  # 0.98
        self.target_steps = self.config.rl.target_steps  # 128
        self.coef_critic_loss = self.config.rl.coef_critic_loss  # 0.5
        self.coef_entropy_loss = self.config.rl.coef_entropy_loss  # 0.01
        self.coef_mask_loss = self.config.rl.coef_mask_loss  # 0.01
        self.lr_scheduler = None
        self.criterion_critic = nn.MSELoss()
        self.compute_advantage_method = kwargs.get('compute_advantage_method', 'gae')
        # train
        self.batch_size = self.config.training.batch_size  # 128
        # tricks
        # self.config.rl.target_kl = kwargs.get('target_kl', 0.01)
        if self.config.rl.norm_reward:
            self.running_stats = RunningMeanStd(shape=1)
        self.config.rl.norm_advantage = kwargs.get('norm_advantage', True)
        self.config.rl.clip_grad = kwargs.get('clip_grad', True)
        self.config.rl.max_grad_norm = kwargs.get('max_grad_norm', 1.)
        self.softmax_temp = 1.
        # log
        self.log_dir = os.path.join(self.save_dir, self.config.logger.log_dir_name)
        self.model_dir = os.path.join(self.save_dir, self.config.training.model_dir_name)
        self.training_info = []
        self.buffer = RolloutBuffer()
        for dir in [self.save_dir, self.log_dir, self.model_dir]:
            if not os.path.exists(dir): 
                os.makedirs(dir)
        # counter
        self.update_time = 0
        self.time_step = 0
        if self.verbose >= 0:
            self.show_config()
        # optimizer
        self.make_policy = make_policy
        self.policy, self.optimizer = self.make_policy(self)
        self.preprocess_obs = obs_as_tensor

    def show_config(self, ):
        print(f'*' * 50)
        print(f'Key parameters of RL training are as following: ')
        print(f'*' * 50)
        print(f'       device: {self.device_name}')
        print(f'   num_workers: {self.num_workers}')
        print(f'  distributed: {self.config.training.distributed_training}')
        print(f'     gamma: {self.config.rl.gamma}')
        print(f'     lr_actor: {self.config.rl.learning_rate.actor}')
        print(f'    lr_critic: {self.config.rl.learning_rate.critic}')
        print(f'   batch_size: {self.batch_size}')
        print(f'coef_ent_loss: {self.coef_entropy_loss}')
        print(f'     norm_adv: {self.config.rl.norm_advantage}')
        print(f'    clip_grad: {self.config.rl.clip_grad}')
        print(f'max_grad_norm: {self.config.rl.max_grad_norm}')
        print(f'save_interval: {self.config.training.save_interval}')
        print(f'eval_interval: {self.config.training.eval_interval}')
        print(f' log_interval: {self.config.training.log_interval}')
        print(f'*' * 50)
        print()
        print(f'Logging training info at {os.path.join(self.log_dir, "training_info.csv")}')

    @abstractmethod
    def preprocess_obs(self, obs):
        raise NotImplementedError

    def solve_with_baseline(self, instance, baseline='grc'):
        if self.baseline_solver is None:
            self.baseline_solver = GRCRankSolver(self.controller, self.recorder, self.counter, self.logger, self.config)
        solution = self.baseline_solver.solve(instance)
        solution_info = self.counter.count_solution(instance['v_net'], solution)
        return solution_info

    def get_action_prob_dist(self, observation):
        with torch.no_grad():
            action_logits = self.policy.act(observation)
        if 'action_mask' in observation and self.config.rl.mask_actions:
            mask = observation['action_mask']
            candidate_action_logits = apply_mask_to_logit(action_logits, mask) 
        else:
            candidate_action_logits = action_logits
        action_prob_dist = F.softmax(candidate_action_logits / self.softmax_temp, dim=-1)
        return action_prob_dist, candidate_action_logits

    def select_action(self, observation, sample=True):
        with torch.no_grad():
            action_logits = self.policy.act(observation)
        if 'action_mask' in observation and self.config.rl.mask_actions:
            mask = observation['action_mask']
            candidate_action_logits = apply_mask_to_logit(action_logits, mask) 
        else:
            candidate_action_logits = action_logits
        candidate_action_dist = Categorical(logits=candidate_action_logits / self.softmax_temp)
        raw_action_dist = Categorical(logits=action_logits / self.softmax_temp)

        if self.config.rl.mask_actions and self.config.rl.maskable_policy:
            action_dist_for_log_prob = candidate_action_dist
        else:
            action_dist_for_log_prob = raw_action_dist

        if sample:
            action = candidate_action_dist.sample()
        else:
            action = candidate_action_logits.argmax(-1)

        action_logprob = action_dist_for_log_prob.log_prob(action)
        
        if torch.numel(action) == 1:
            action = action.item()
        else:
            action = action.reshape(-1, ).cpu().detach().numpy()
        # action = action.squeeze(-1).cpu()
        return action, action_logprob.cpu().detach().numpy()

    def evaluate_actions(self, old_observations, old_actions, return_others=False):
        actions_logits = self.policy.act(old_observations)
        actions_probs = F.softmax(actions_logits / self.softmax_temp, dim=-1)
        if 'action_mask' in old_observations:
            masks = old_observations['action_mask']
            candidate_actions_logits = apply_mask_to_logit(actions_logits, masks)
        else:
            masks = None
            candidate_actions_logits = actions_logits

        candidate_actions_probs = F.softmax(candidate_actions_logits, dim=-1)

        dist = Categorical(actions_probs)
        candidate_dist = Categorical(candidate_actions_probs)
        policy_dist = candidate_dist if self.config.rl.mask_actions and self.config.rl.maskable_policy else dist

        action_logprobs = policy_dist.log_prob(old_actions)
        dist_entropy = policy_dist.entropy()

        values = self.policy.evaluate(old_observations).squeeze(-1) if hasattr(self.policy, 'evaluate') else None

        if return_others:
            other = {}
            if masks is not None:
                mask_actions_probs = actions_probs * (~masks.bool())
                other['mask_actions_probs'] = mask_actions_probs.sum(-1).mean()
                if hasattr(self.policy, 'predictor'):
                    predicted_masks_logits = self.policy.predict(old_observations)
                    print(predicted_masks_logits)
                    prediction_loss = F.binary_cross_entropy(predicted_masks_logits, masks.float())
                    other['prediction_loss'] = prediction_loss
                    predicted_masks = torch.where(predicted_masks_logits > 0.5, 1., 0.)
                    correct_count = torch.eq(predicted_masks.bool(), masks.bool()).sum(-1).float().mean(0)
                    acc = correct_count / predicted_masks.shape[-1]
                    print(prediction_loss, correct_count, acc)
            return values, action_logprobs, dist_entropy, other

        return values, action_logprobs, dist_entropy

    def estimate_value(self, observation):
        """
        Estimate the value of an observation
        """
        with torch.no_grad():
            estimated_value = self.policy.evaluate(observation).squeeze(-1).detach().cpu().item()
        return estimated_value

    def calculate_fixed_advantages(self, returns, rollout_values):
        """Calculate policy advantages from values frozen during rollout collection."""
        if len(rollout_values) != returns.numel():
            raise ValueError(
                'Expected one rollout value per return, but got '
                f'{len(rollout_values)} values and {returns.numel()} returns.'
            )
        old_values = torch.tensor(
            [
                float(value.detach().cpu().reshape(-1)[0])
                if isinstance(value, torch.Tensor)
                else float(value)
                for value in rollout_values
            ],
            dtype=returns.dtype,
            device=returns.device,
        )
        return returns - old_values

    def normalize_advantages(self, advantages):
        if self.config.rl.norm_advantage and advantages.numel() > 1:
            return (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-9)
        return advantages

    def calculate_update_sample_times(self):
        """Return the mini-batch updates needed for the configured PPO epochs."""
        if self.buffer.size() <= 0:
            raise ValueError('Cannot update a policy from an empty rollout buffer.')
        if self.batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {self.batch_size}.')
        if self.repeat_times < 0:
            raise ValueError(f'repeat_times must be non-negative, got {self.repeat_times}.')
        target_samples = self.buffer.size() * self.repeat_times
        return max(1, math.ceil(target_samples / self.batch_size))

    def to_sub_solver(self):
        temp_dict = {}
        unnecessary_attributes = ['policy', 'optimizer', 'lr_scheduler', 'searcher', 'writer', 'logger']
        for attr_name in unnecessary_attributes:
            if hasattr(self, attr_name):
                temp_dict[attr_name] = getattr(self, attr_name)
                delattr(self, attr_name)
        sub_solver = copy.deepcopy(self)
        sub_solver.policy, _ = sub_solver.make_policy(sub_solver)
        sub_solver.logger = None
        for attr_name in temp_dict.keys():
            setattr(self, attr_name, temp_dict[attr_name])
        sub_solver.policy.load_state_dict(self.policy.state_dict())
        sub_solver.policy.to(self.device)
        return sub_solver

    def get_worker(self, rank=0):
        self.policy.share_memory()
        shared_policy = self.policy
        self.optimizer = SharedAdam.from_optim(self.optimizer)
        shared_optimizer = self.optimizer

        lr_scheduler = self.lr_scheduler
        writer = self.writer
        unnecessary_attributes = ['policy', 'optimizer', 'lr_scheduler', 'searcher', 'writer']
        for attr_name in unnecessary_attributes:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        # self.eval()
        worker = copy.deepcopy(self)
        worker.rank = rank
        worker.writer = None
        worker.lr_scheduler = None
        import pdb; pdb.set_trace()
        if self.use_cuda:
            num_gpu_devices = torch.cuda.device_count()
            worker_device_id = rank % num_gpu_devices
            worker.device = torch.device(f'cuda:{worker_device_id}')
        worker.policy, worker.optimizer = worker.make_policy(worker)
        worker.policy.load_state_dict(shared_policy.state_dict())
        worker.policy.to(worker.device)
        # worker.optimizer = torch.optim.Adam(worker.policy.parameters(), lr=self.lr)
        worker.shared_policy = shared_policy
        worker.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.policy = shared_policy
        self.optimizer = shared_optimizer
        return worker

    def save_model(self, checkpoint_fname):
        checkpoint_fname = os.path.join(self.model_dir, checkpoint_fname)
        checkpoint = {
            'checkpoint_version': 2,
            'feature_schema_version': int(OmegaConf.select(
                self.config,
                'rl.feature_constructor.schema_version',
                default=1,
            )),
            'solver_name': self.config.solver.solver_name,
            'p_net_num_nodes': OmegaConf.select(
                self.config,
                'simulation.p_net_setting_num_nodes',
            ),
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }
        checkpoint.update(self.get_additional_checkpoint_state())
        torch.save(checkpoint, checkpoint_fname)
        self.logger.critical(f'Save model to {checkpoint_fname}\n')

    def get_additional_checkpoint_state(self):
        """Return algorithm-specific state stored alongside the main policy."""
        return {}

    def load_additional_checkpoint_state(self, checkpoint):
        """Restore algorithm-specific state from a checkpoint."""
        return None

    def load_model(self, checkpoint_path):
        print('Attempting to load the pretrained model')
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            if not isinstance(checkpoint, Mapping):
                raise TypeError(
                    f'Expected a state-dict checkpoint, got {type(checkpoint).__name__}'
                )

            if 'policy' in checkpoint:
                policy_state = checkpoint['policy']
                optimizer_state = checkpoint.get('optimizer')
                feature_schema_version = checkpoint.get('feature_schema_version')
            else:
                policy_state = checkpoint
                optimizer_state = None
                feature_schema_version = None

            self.policy.load_state_dict(policy_state, strict=True)
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            self.load_additional_checkpoint_state(checkpoint)

            if feature_schema_version is None:
                feature_schema_version = 1
                self.logger.warning(
                    'Checkpoint has no feature schema metadata; using legacy '
                    'feature schema v1 for compatibility.'
                )
            if int(feature_schema_version) not in (1, 2):
                raise ValueError(
                    'Unsupported checkpoint feature schema version: '
                    f'{feature_schema_version}'
                )
            with open_dict(self.config):
                self.config.rl.feature_constructor.schema_version = int(
                    feature_schema_version
                )
            self.logger.critical(f'Parameter Initialization: Loaded pretrained model from {checkpoint_path}')
        except Exception as e:
            message = (
                f'Parameter Initialization: Load pretrained failed from '
                f'{checkpoint_path}\n{e}'
            )
            self.logger.critical(message)
            raise RuntimeError(message) from e

    def train(self):
        """Set the mode to train"""
        self.policy.train()
        if hasattr(self, 'searcher'):
            delattr(self, 'searcher')

    def eval(self, decode_strategy=None, k=None):
        if decode_strategy is None:
            decode_strategy = self.config.inference.decode_strategy
        if k is None:
            k = self.config.inference.k_searching
        assert k >= 1, f'k should greater than 0. (k={k})'
        self.policy.eval()
        self.searcher = get_searcher(decode_strategy, 
                                    policy=self.policy, 
                                    preprocess_obs_func=self.preprocess_obs, 
                                    make_policy_func=self.make_policy,
                                    k=k, device=self.device,
                                    mask_actions=self.config.rl.mask_actions, 
                                    maskable_policy=self.config.rl.maskable_policy)

    def update_grad(self, loss):
        # update parameters
        if self.config.training.distributed_training:
            with self.lock:
                self.optimizer.zero_grad()
                self.shared_optimizer.zero_grad()
                loss.backward()
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm) if self.config.rl.clip_grad else None
                sync_gradients(self.shared_policy, self.policy)
                self.optimizer.step()
                self.shared_optimizer.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm) if self.config.rl.clip_grad else None
            self.optimizer.step()
        return grad_clipped

    def sync_parameters(self):
        assert self.config.training.distributed_training, 'distributed_training should be True'
        with self.lock:
            self.policy.load_state_dict(self.shared_policy.state_dict())

    def learn(self, env, num_epochs=1, **kwargs):
        self.start_time = time.time()
        if self.config.training.distributed_training:
            self.learn_distributedly(env, num_epochs)
        else:
            self.logger.info(f'Start to learn singly')
            self.learn_singly(env, num_epochs)
        print(f'Start to validate')
        self.save_model(f'model.pkl')
        # self.validate(env)
        self.end_time = time.time()
        print(f'\nTotal training time: {(self.end_time - self.start_time) / 3600:4.6f} h')

    def learn_distributedly(self, env, num_epochs, **kwargs):
        assert self.config.training.distributed_training, 'distributed_training should be True'
        assert num_epochs % self.num_workers == 0, 'num_epochs should be divisible by num_workers'
        job_list = []
        mp.set_start_method('spawn')
        lock = mp.Lock()
        worker_num_epochs = int(num_epochs // self.num_workers)
        worker_save_interval = int(np.ceil(self.config.training.save_interval / self.num_workers))
        worker_eval_interval = int(np.ceil(self.config.training.eval_interval / self.num_workers))
        print(f'Distributed training with {self.num_workers} workers')
        print(f'Worker num epochs:    {   worker_num_epochs:3d} epochs')
        print(f'Worker save interval: {worker_save_interval:3d} epochs')
        print(f'Worker eval interval: {worker_eval_interval:3d} epochs')
        print()
        for worker_rank in range(self.num_workers):
            env = copy.deepcopy(env)
            worker = self.get_worker(worker_rank)
            worker.lock = lock
            worker.save_interval = worker_save_interval
            worker.eval_interval = worker_eval_interval
            if worker_rank != 0: 
                worker.verbose = 0
                env.verbose = 0
            job = mp.Process(target=worker.learn_singly, args=(env, worker_num_epochs))
            job_list.append(job)
            job.start()
        for job in job_list: 
            job.join()


class PGSolver(RLSolver):
    
    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(PGSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)

    def update(self, ):
        observations = self.preprocess_obs(self.buffer.observations, self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        returns = torch.FloatTensor(np.array(self.buffer.returns)).to(self.device)
        _, action_logprobs, _, _ = self.evaluate_actions(observations, actions, return_others=True)
        
        loss = - (action_logprobs * returns).mean()

        grad_clipped = self.update_grad(loss)

        info = {
            'lr': self.optimizer.defaults['lr'],
            'loss/loss': loss.detach().cpu(),
            'value/logprob': action_logprobs.detach().mean().cpu(),
            'value/return': returns.detach().mean().cpu(),
        }
        self.logger.log(data=info, step=self.update_time)

        self.buffer.clear()
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        self.update_time += 1

        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()


class A2CSolver(RLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(A2CSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.repeat_times = 1

    def update(self, ):
        observations = self.preprocess_obs(self.buffer.observations, self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
        advantages = returns - values.detach()
        if self.config.rl.norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = - (action_logprobs * advantages).mean()
        critic_loss = F.mse_loss(returns, values)
        entropy_loss = dist_entropy.mean()
        loss = actor_loss + self.coef_critic_loss * critic_loss - self.coef_entropy_loss * entropy_loss

        grad_clipped = self.update_grad(loss)

        info = {
            'lr': self.optimizer.defaults['lr'],
            'loss/loss': loss.detach().cpu().numpy(),
            'loss/actor_loss': actor_loss.detach().cpu().numpy(),
            'loss/critic_loss': critic_loss.detach().cpu().numpy(),
            'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
            'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
            'value/return': returns.detach().mean().cpu().numpy()
        }
        self.logger.log(data=info, step=self.update_time)

        self.buffer.clear()
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        self.update_time += 1

        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()


class PPOSolver(RLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(PPOSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.repeat_times = kwargs.get('repeat_times', 10)
        self.gae_lambda = kwargs.get('gae_lambda', 0.98)
        self.eps_clip = kwargs.get('eps_clip', 0.2)

    def calculate_kl_divergence(self, old_observations, old_policy):
        actions_logits = self.policy.act(old_observations)
        old_actions_logits = old_policy.act(old_observations)
        actions_probs = F.softmax(actions_logits / self.softmax_temp, dim=-1)
        old_actions_probs = F.softmax(old_actions_logits / self.softmax_temp, dim=-1)
        dist = Categorical(actions_probs)
        old_dist = Categorical(old_actions_probs)
        kl_divergence = torch.distributions.kl_divergence(dist, old_dist).mean()
        return kl_divergence
        

    def update(self, ):
        # assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')

        batch_observations = self.preprocess_obs(self.buffer.observations, device)
        # # batch_actions = torch.LongTensor(np.concatenate(self.buffer.actions, axis=0)).to(self.device)
        batch_actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_returns = torch.FloatTensor(self.buffer.returns)
        if self.config.rl.norm_reward:
            batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-9)
        batch_advantages = self.calculate_fixed_advantages(
            batch_returns,
            self.buffer.values,
        )
        sample_times = self.calculate_update_sample_times()
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            # observations  = get_observations_sample(batch_observations, sample_indices, self.device)
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            advantages = self.normalize_advantages(
                batch_advantages[sample_indices].to(self.device)
            )
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = self.criterion_critic(returns, values)
            entropy_loss = dist_entropy.mean()
            approx_kl = (old_action_logprobs - action_logprobs).mean().detach().cpu().numpy()

            mask_loss = other.get('mask_actions_probs', 0)
            prediction_loss = other.get('prediction_loss', 0)

            loss = actor_loss + self.coef_critic_loss * critic_loss - self.coef_entropy_loss * entropy_loss + self.coef_mask_loss * mask_loss + prediction_loss

            if self.config.rl.target_kl is not None and approx_kl > 1.5 * self.config.rl.target_kl:
                print(f'Early stopping at update time {self.update_time}, kl: {approx_kl:2.4f}')
                break

            # update parameters
            grad_clipped = self.update_grad(loss)
        
            if self.update_time % self.config.training.log_interval == 0:
                info = {
                    # 'lr': self.optimizer.defaults['lr'],
                    'loss/loss': loss.detach().cpu().numpy(),
                    'loss/actor_loss': actor_loss.detach().cpu().numpy(),
                    'loss/critic_loss': critic_loss.detach().cpu().numpy(),
                    'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
                    'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
                    'value/old_action_logprob': old_action_logprobs.mean().cpu().numpy(),
                    'info/approx_kl': approx_kl,
                    'value/value': values.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/reward': batch_rewards.mean().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                self.logger.log(data=info, step=self.update_time)

            self.update_time += 1

        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()

        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()


class A3CSolver(PPOSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(A3CSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)


class ARPPOSolver(RLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(ARPPOSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.repeat_times = kwargs.get('repeat_times', 10)
        self.gae_lambda = kwargs.get('gae_lambda', 0.98)
        self.eps_clip = kwargs.get('eps_clip', 0.2)

    def update(self, ):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        mean_batch_rewards = batch_rewards.mean()

        batch_returns = torch.FloatTensor(self.buffer.returns)
        batch_advantages = self.calculate_fixed_advantages(
            batch_returns,
            self.buffer.values,
        )
        sample_times = self.calculate_update_sample_times()
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            observations = get_observations_sample(batch_observations, sample_indices, device=self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            advantages = self.normalize_advantages(
                batch_advantages[sample_indices].to(self.device)
            )
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # masks = batch_masks[sample_indices].to(self.device) if batch_masks is not None else None
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = self.criterion_critic(returns, values)
            entropy_loss = dist_entropy.mean()

            mask_loss = other.get('mask_actions_probs', 0)
            prediction_loss = other.get('prediction_loss', 0)

            loss = actor_loss + self.coef_critic_loss * critic_loss - self.coef_entropy_loss * entropy_loss + self.coef_mask_loss * mask_loss + prediction_loss
            # update parameters
            grad_clipped = self.update_grad(loss)
    
            if self.update_time % self.config.training.log_interval == 0:
                info = {
                    'lr': self.optimizer.defaults['lr'],
                    'loss/loss': loss.detach().cpu().numpy(),
                    'loss/actor_loss': actor_loss.detach().cpu().numpy(),
                    'loss/critic_loss': critic_loss.detach().cpu().numpy(),
                    'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
                    'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
                    'value/old_action_logprob': old_action_logprobs.mean().cpu().numpy(),
                    'value/value': values.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'value/advantages': advantages.mean().cpu().numpy(),
                    'value/reward': batch_rewards.mean().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                self.logger.log(data=info, step=self.update_time)

            self.update_time += 1

        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        self.buffer.clear()

        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()


class DPGSolver(RLSolver):
    """Placeholder for continuous deterministic policy gradients."""


class _DiscreteOffPolicySolver(RLSolver):
    """Shared replay and exploration mechanics for discrete off-policy RL."""

    algorithm_name = None

    def __init__(self, controller, recorder, counter, logger, config,
                 make_policy, obs_as_tensor, **kwargs):
        super().__init__(
            controller,
            recorder,
            counter,
            logger,
            config,
            make_policy,
            obs_as_tensor,
            **kwargs,
        )
        if self.config.training.distributed_training:
            raise NotImplementedError(
                'DQN and discrete DDPG do not support distributed training.'
            )

        self.is_off_policy = True
        self.replay_capacity = int(self._parameter(
            'replay_capacity', 10000, kwargs
        ))
        self.learning_starts = int(self._parameter(
            'learning_starts', self.batch_size, kwargs
        ))
        self.gradient_steps_per_transition = int(self._parameter(
            'gradient_steps_per_transition', 1, kwargs
        ))
        self.epsilon_start = float(self._parameter(
            'epsilon_start', 1.0, kwargs
        ))
        self.epsilon_end = float(self._parameter(
            'epsilon_end', 0.1, kwargs
        ))
        self.epsilon_decay = float(self._parameter(
            'epsilon_decay', 10000, kwargs
        ))
        if self.replay_capacity < self.batch_size:
            raise ValueError(
                'replay_capacity must be at least batch_size, got '
                f'{self.replay_capacity} < {self.batch_size}.'
            )
        if self.learning_starts < 0:
            raise ValueError(
                f'learning_starts must be non-negative, got {self.learning_starts}.'
            )
        if self.gradient_steps_per_transition <= 0:
            raise ValueError('gradient_steps_per_transition must be positive.')
        if self.epsilon_decay <= 0:
            raise ValueError(f'epsilon_decay must be positive, got {self.epsilon_decay}.')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError(
                'Expected 0 <= epsilon_end <= epsilon_start <= 1, got '
                f'{self.epsilon_end} and {self.epsilon_start}.'
            )

        self.target_steps = max(self.batch_size, self.learning_starts)
        self.steps_done = 0
        self.epsilon = self.epsilon_start

    def _parameter(self, name, default, kwargs):
        if name in kwargs:
            return kwargs[name]
        return OmegaConf.select(
            self.config,
            f'rl.{self.algorithm_name}.{name}',
            default=default,
        )

    def _calc_epsilon(self):
        return float(
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * np.exp(-float(self.steps_done) / self.epsilon_decay)
        )

    def _masked_scores(self, scores, observation):
        if self.config.rl.mask_actions and 'action_mask' in observation:
            return apply_mask_to_logit(scores, observation['action_mask'])
        return scores

    def _greedy_action(self, observation):
        with torch.no_grad():
            scores = self._masked_scores(
                self.policy.act(observation), observation
            )
        return int(scores.argmax(dim=-1).reshape(-1)[0].item())

    def _random_action(self, observation):
        if self.config.rl.mask_actions and 'action_mask' in observation:
            mask = observation['action_mask'].detach().bool().reshape(-1)
            candidates = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if candidates.numel() == 0:
                raise ValueError('Cannot sample from an empty action mask.')
            index = torch.randint(candidates.numel(), (1,), device=candidates.device)
            return int(candidates[index].item())
        num_actions = self.policy.act(observation).shape[-1]
        return int(np.random.randint(0, num_actions))

    def select_action(self, observation, sample=True):
        if sample:
            self.steps_done += 1
            self.epsilon = self._calc_epsilon()
            action = self._random_action(observation) \
                if np.random.random() < self.epsilon \
                else self._greedy_action(observation)
        else:
            action = self._greedy_action(observation)
        return action, np.zeros((1, 1), dtype=np.float32)

    def get_update_steps(self, instance_buffer):
        return max(
            1,
            instance_buffer.size() * self.gradient_steps_per_transition,
        )

    def _sample_replay(self):
        if self.buffer.size() < self.batch_size:
            raise ValueError(
                f'Need at least {self.batch_size} replay transitions, got '
                f'{self.buffer.size()}.'
            )
        indices = torch.randint(
            self.buffer.size(), size=(self.batch_size,)
        ).tolist()
        observations = self.preprocess_obs(
            [self.buffer.observations[index] for index in indices],
            self.device,
        )
        next_observations = self.preprocess_obs(
            [self.buffer.next_observations[index] for index in indices],
            self.device,
        )
        actions = torch.as_tensor(
            [self.buffer.actions[index] for index in indices],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [self.buffer.rewards[index] for index in indices],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [self.buffer.dones[index] for index in indices],
            dtype=torch.float32,
            device=self.device,
        )
        return observations, actions, rewards, dones, next_observations

    @staticmethod
    def _freeze_target(network):
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)


class DQNSolver(_DiscreteOffPolicySolver):
    """Deep Q-Network with persistent replay and a hard target network."""

    algorithm_name = 'dqn'

    def __init__(self, controller, recorder, counter, logger, config,
                 make_policy, obs_as_tensor, **kwargs):
        super().__init__(
            controller,
            recorder,
            counter,
            logger,
            config,
            make_policy,
            obs_as_tensor,
            **kwargs,
        )
        self.target_update_interval = int(self._parameter(
            'target_update_interval', 100, kwargs
        ))
        if self.target_update_interval <= 0:
            raise ValueError('target_update_interval must be positive.')
        self.target_policy = copy.deepcopy(self.policy).to(self.device)
        self._freeze_target(self.target_policy)

    def _next_q_values(self, next_observations):
        target_scores = self._masked_scores(
            self.target_policy.act(next_observations), next_observations
        )
        return target_scores.max(dim=-1).values

    def update(self):
        observations, actions, rewards, dones, next_observations = \
            self._sample_replay()
        q_values = self.policy.act(observations).gather(
            1, actions.unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            next_q_values = self._next_q_values(next_observations)
            target_q_values = rewards + (
                self.config.rl.gamma * next_q_values * (1.0 - dones)
            )

        loss = F.smooth_l1_loss(q_values, target_q_values)
        grad_clipped = self.update_grad(loss)
        self.update_time += 1
        if self.update_time % self.target_update_interval == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        if self.update_time % self.config.training.log_interval == 0:
            self.logger.log(data={
                'loss/q_loss': loss.item(),
                'value/q_value': q_values.detach().mean().item(),
                'value/target_q_value': target_q_values.mean().item(),
                'value/reward': rewards.mean().item(),
                'value/done': dones.mean().item(),
                'value/epsilon': self.epsilon,
                'grad/grad_clipped': None if grad_clipped is None
                else grad_clipped.detach().cpu().item(),
                'lr': self.optimizer.param_groups[0]['lr'],
            }, step=self.update_time)
        return loss.detach()

    def train(self):
        super().train()
        self.target_policy.eval()

    def get_additional_checkpoint_state(self):
        return {
            'target_policy': self.target_policy.state_dict(),
            'steps_done': int(self.steps_done),
            'epsilon': float(self.epsilon),
            'update_time': int(self.update_time),
        }

    def load_additional_checkpoint_state(self, checkpoint):
        target_state = checkpoint.get('target_policy')
        self.target_policy.load_state_dict(
            self.policy.state_dict() if target_state is None else target_state
        )
        self.steps_done = int(checkpoint.get('steps_done', 0))
        self.epsilon = float(checkpoint.get('epsilon', self._calc_epsilon()))
        self.update_time = int(checkpoint.get('update_time', 0))


class DoubleDQNSolver(DQNSolver):
    """Double DQN target selection using the online Q-network."""

    def _next_q_values(self, next_observations):
        online_scores = self._masked_scores(
            self.policy.act(next_observations), next_observations
        )
        next_actions = online_scores.argmax(dim=-1, keepdim=True)
        return self.target_policy.act(next_observations).gather(
            1, next_actions
        ).squeeze(-1)


class DDPGSolver(_DiscreteOffPolicySolver):
    """DDPG-style actor-critic adapted to Virne's discrete node actions.

    Vanilla DDPG assumes continuous actions. Here the actor emits masked node
    logits, the critic estimates one Q-value per node, and the actor maximizes
    the critic's expected Q under a differentiable categorical relaxation.
    Both networks use slowly updated target copies and persistent replay.
    """

    algorithm_name = 'ddpg'

    def __init__(self, controller, recorder, counter, logger, config,
                 make_policy, obs_as_tensor, **kwargs):
        super().__init__(
            controller,
            recorder,
            counter,
            logger,
            config,
            make_policy,
            obs_as_tensor,
            **kwargs,
        )
        self.tau = float(self._parameter('tau', 0.005, kwargs))
        self.policy_temperature = float(self._parameter(
            'policy_temperature', 1.0, kwargs
        ))
        if not 0 < self.tau <= 1:
            raise ValueError(f'tau must be in (0, 1], got {self.tau}.')
        if self.policy_temperature <= 0:
            raise ValueError('policy_temperature must be positive.')

        self.critic_policy, _ = self.make_policy(self)
        self.critic_policy.to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_policy.parameters(),
            lr=self.config.rl.learning_rate.critic,
            weight_decay=self.config.rl.weight_decay,
        )
        self.target_policy = copy.deepcopy(self.policy).to(self.device)
        self.target_critic_policy = copy.deepcopy(
            self.critic_policy
        ).to(self.device)
        self._freeze_target(self.target_policy)
        self._freeze_target(self.target_critic_policy)

    @staticmethod
    def _soft_update(target, source, tau):
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                    target.parameters(), source.parameters()):
                target_parameter.lerp_(source_parameter, tau)
            for target_buffer, source_buffer in zip(
                    target.buffers(), source.buffers()):
                target_buffer.copy_(source_buffer)

    def update(self):
        observations, actions, rewards, dones, next_observations = \
            self._sample_replay()

        current_q_values = self.critic_policy.act(observations).gather(
            1, actions.unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            target_actor_scores = self._masked_scores(
                self.target_policy.act(next_observations), next_observations
            )
            next_actions = target_actor_scores.argmax(dim=-1, keepdim=True)
            next_q_values = self.target_critic_policy.act(
                next_observations
            ).gather(1, next_actions).squeeze(-1)
            target_q_values = rewards + (
                self.config.rl.gamma * next_q_values * (1.0 - dones)
            )

        critic_loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_clipped = torch.nn.utils.clip_grad_norm_(
            self.critic_policy.parameters(), self.config.rl.max_grad_norm
        ) if self.config.rl.clip_grad else None
        self.critic_optimizer.step()

        actor_scores = self._masked_scores(
            self.policy.act(observations), observations
        )
        action_probabilities = F.softmax(
            actor_scores / self.policy_temperature, dim=-1
        )
        with torch.no_grad():
            critic_scores = self.critic_policy.act(observations)
        actor_loss = -(
            action_probabilities * critic_scores
        ).sum(dim=-1).mean()
        actor_grad_clipped = self.update_grad(actor_loss)

        self._soft_update(self.target_policy, self.policy, self.tau)
        self._soft_update(
            self.target_critic_policy, self.critic_policy, self.tau
        )
        self.update_time += 1

        if self.update_time % self.config.training.log_interval == 0:
            self.logger.log(data={
                'loss/actor_loss': actor_loss.detach().item(),
                'loss/critic_loss': critic_loss.detach().item(),
                'value/q_value': current_q_values.detach().mean().item(),
                'value/target_q_value': target_q_values.mean().item(),
                'value/reward': rewards.mean().item(),
                'value/done': dones.mean().item(),
                'value/epsilon': self.epsilon,
                'grad/actor_grad_clipped': None
                if actor_grad_clipped is None
                else actor_grad_clipped.detach().cpu().item(),
                'grad/critic_grad_clipped': None
                if critic_grad_clipped is None
                else critic_grad_clipped.detach().cpu().item(),
                'lr': self.optimizer.param_groups[0]['lr'],
            }, step=self.update_time)
        return (actor_loss + critic_loss).detach()

    def train(self):
        super().train()
        self.critic_policy.train()
        self.target_policy.eval()
        self.target_critic_policy.eval()

    def get_additional_checkpoint_state(self):
        return {
            'critic_policy': self.critic_policy.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'target_policy': self.target_policy.state_dict(),
            'target_critic_policy': self.target_critic_policy.state_dict(),
            'steps_done': int(self.steps_done),
            'epsilon': float(self.epsilon),
            'update_time': int(self.update_time),
        }

    def load_additional_checkpoint_state(self, checkpoint):
        critic_state = checkpoint.get('critic_policy')
        if critic_state is None:
            raise ValueError('Discrete DDPG checkpoint is missing critic_policy.')
        self.critic_policy.load_state_dict(critic_state)
        if 'critic_optimizer' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.target_policy.load_state_dict(
            checkpoint.get('target_policy', self.policy.state_dict())
        )
        self.target_critic_policy.load_state_dict(
            checkpoint.get('target_critic_policy', critic_state)
        )
        self.steps_done = int(checkpoint.get('steps_done', 0))
        self.epsilon = float(checkpoint.get('epsilon', self._calc_epsilon()))
        self.update_time = int(checkpoint.get('update_time', 0))
