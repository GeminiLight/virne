# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import csv
import copy
import time
from omegaconf import open_dict
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
        # torch.save(self.policy.state_dict(), checkpoint_fname)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }, checkpoint_fname)
        self.logger.critical(f'Save model to {checkpoint_fname}\n')

    def load_model(self, checkpoint_path):
        print('Attempting to load the pretrained model')
        try:
            checkpoint = torch.load(checkpoint_path)
            if 'policy' not in checkpoint:
                self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
            else:
                self.policy.load_state_dict(checkpoint['policy'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.critical(f'Parameter Initialization: Loaded pretrained model from {checkpoint_path}')
        except Exception as e:
            self.logger.critical(f'Parameter Initialization: Load pretrained failed from {checkpoint_path}\n{e}\nInitilized with random parameters')

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
        loss = actor_loss + self.coef_critic_loss * critic_loss + self.coef_entropy_loss * entropy_loss

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
        # copy the old policy parameters
        old_policy = copy.deepcopy(self.policy)

        batch_observations = self.preprocess_obs(self.buffer.observations, device)
        # # batch_actions = torch.LongTensor(np.concatenate(self.buffer.actions, axis=0)).to(self.device)
        batch_actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_returns = torch.FloatTensor(self.buffer.returns)
        if self.config.rl.norm_reward:
            batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-9)
        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            # observations  = get_observations_sample(batch_observations, sample_indices, self.device)
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            
            # calculate advantage
            advantages = returns - values.detach()
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
  
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
        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            observations = get_observations_sample(batch_observations, sample_indices, device=self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # masks = batch_masks[sample_indices].to(self.device) if batch_masks is not None else None
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            
            # calculate advantage
            advantages = returns - values.detach()
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
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

    """

    """
    


class DDPGSolver(RLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(DDPGSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.repeat_times = kwargs.get('repeat_times', 10)
        self.gae_lambda = kwargs.get('gae_lambda', 0.98)
        self.eps_clip = kwargs.get('eps_clip', 0.2)

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
        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            # observations  = get_observations_sample(batch_observations, sample_indices, self.device)
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            
            # calculate advantage
            advantages = returns - values.detach()
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
  
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


class DQNSolver(RLSolver):
    
    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(DQNSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        # DQN-specific parameters
        self.target_update_interval = kwargs.get('target_update_interval', 100)
        self.epsilon_start = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.1)
        self.epsilon_decay = kwargs.get('epsilon_decay', 10000)
        self.buffer_size = kwargs.get('buffer_size', 10000)
        self.batch_size = kwargs.get('batch_size', 64)
        self.config.rl.gamma = kwargs.get('gamma', 0.99)
        self.lr = kwargs.get('lr', 0.001)
        
        self.policy, self.optimizer = self.make_policy(self)
        self.target_policy = copy.deepcopy(self.policy)
        self.replay_buffer = RolloutBuffer()
        self.epsilon = self.epsilon_start
        self.steps_done = 0
        self.target_steps = self.batch_size

    def _calc_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            np.exp(-1. * (self.steps_done) / self.epsilon_decay)

    def select_action(self, observation, sample=True):
        """
        Epsilon-greedy action selection.
        """
        def greedy_action(observation):
            with torch.no_grad():
                action_q_values = self.policy.act(observation)
                if 'action_mask' in observation and self.config.rl.mask_actions and self.config.rl.maskable_policy:
                    action_q_values = apply_mask_to_logit(action_q_values, observation['action_mask'])
                action = action_q_values.argmax().item()
            return action

        if sample:
            self.steps_done += 1
            self.epsilon = self._calc_epsilon()
            if np.random.rand() < self.epsilon:
                # Random action
                action = np.random.randint(0, self.p_net_setting_num_nodes)
            else:
                action = greedy_action(observation)
        else:
            action = greedy_action(observation)

        zero_logprob = np.zeros((1, 1), dtype=np.float32)
        return action, zero_logprob


    def update(self):
        """
        Update the Q-network based on a batch of experiences.
        """
        # if self.replay_buffer.size() < self.batch_size:
            # return
        batch_actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_dones = torch.FloatTensor(self.buffer.dones)
        sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
        # observations  = get_observations_sample(batch_observations, sample_indices, self.device)
        sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
        observations = self.preprocess_obs(sample_obersevations, self.device)
        sample_next_obersevations = [self.buffer.next_observations[i] for i in sample_indices]
        next_observations = self.preprocess_obs(sample_next_obersevations, self.device)
        actions = batch_actions[sample_indices].to(self.device)
        rewards = batch_rewards[sample_indices].to(self.device)
        dones = batch_dones[sample_indices].to(self.device)

        # Compute current Q values
        q_values = self.policy.act(observations).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_policy.act(next_observations).max(dim=1)[0]
            target_q_values = rewards + self.config.rl.gamma * max_next_q_values * (1 - dones)
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
        self.optimizer.step()

        # Periodically update the target network
        if self.update_time % self.target_update_interval == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        self.update_time += 1
        self.logger.log({
            'loss/q_loss': loss.item(),
            'value/q_value': q_values.mean().item(),
            'value/target_q_value': target_q_values.mean().item(),
            'value/reward': rewards.mean().item(),
            'value/return': batch_rewards.mean().item(),
            'value/done': dones.mean().item(),
            'value/eplison': self.epsilon,
            'lr': self.optimizer.defaults['lr'],
        }, self.update_time)

        self.buffer.clear()
        return loss.detach()



class DoubleDQNSolver(DQNSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(DoubleDQNSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_steps = self.target_update_interval

    def update(self):
        """
        Update the Q-network based on a batch of experiences.
        """
        # if self.replay_buffer.size() < self.batch_size:
            # return
        batch_actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_dones = torch.FloatTensor(self.buffer.dones)
        sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
        # observations  = get_observations_sample(batch_observations, sample_indices, self.device)
        sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
        observations = self.preprocess_obs(sample_obersevations, self.device)
        sample_next_obersevations = [self.buffer.next_observations[i] for i in sample_indices]
        next_observations = self.preprocess_obs(sample_next_obersevations, self.device)
        actions = batch_actions[sample_indices].to(self.device)
        rewards = batch_rewards[sample_indices].to(self.device)
        dones = batch_dones[sample_indices].to(self.device)

        # Compute current Q values
        q_values = self.policy.act(observations).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q values
        with torch.no_grad():
            max_next_actions = self.policy.act(next_observations).max(dim=1)[1]
            max_next_q_values = self.target_policy.act(next_observations).gather(1, max_next_actions.unsqueeze(-1)).squeeze(-1)
            target_q_values = rewards + self.config.rl.gamma * max_next_q_values * (1 - dones)
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
        self.optimizer.step()

        # Periodically update the target network
        if self.update_time % self.target_update_interval == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        self.update_time += 1
        self.logger.log({
            'loss/q_loss': loss.item(),
            'value/q_value': q_values.mean().item(),
            'value/target_q_value': target_q_values.mean().item(),
            'value/reward': rewards.mean().item(),
            'value/return': batch_rewards.mean().item(),
            'value/done': dones.mean().item(),
            'value/eplison': self.epsilon,
            'lr': self.optimizer.defaults['lr'],
        }, self.update_time)

        self.buffer.clear()
        return loss.detach()