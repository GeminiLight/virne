# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import csv
import copy
import time
import tqdm
import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pool
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs):
        super(RLSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.rank = 0
        # baseline
        self.use_baseline_solver = kwargs.get('use_baseline_solver', False)
        if self.use_baseline_solver:
            self.baseline_solvers = {}
            baselin_solver_name =  kwargs.get('baselin_solver_name', 'grc')
            if baselin_solver_name == 'grc':
                self.baseline_solver = GRCRankSolver(controller, recorder, counter, **kwargs)
            elif baselin_solver_name == 'self':
                self.baseline_solver = copy.deepcopy(self)
        # training
        self.use_cuda = kwargs.get('use_cuda', True)
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            self.device = torch.device('cpu')
            self.device_name = 'CPU'
            self.use_cuda = False
        self.num_workers = kwargs.get('num_workers', 1)
        self.distributed_training = False if self.num_workers == 1 else kwargs.get('distributed_training', False)
        # rl
        self.gamma = kwargs.get('rl_gamma', 0.95)
        self.gae_lambda = kwargs.get('gae_lambda', 0.98)
        self.coef_critic_loss = kwargs.get('coef_critic_loss', 0.5)
        self.coef_entropy_loss = kwargs.get('coef_entropy_loss', 0.01)
        self.coef_mask_loss = kwargs.get('coef_mask_loss', 0.01)
        self.weight_decay = kwargs.get('weight_decay', 0.00001)
        self.lr = kwargs.get('lr', 0.001)
        self.lr_actor = kwargs.get('lr_actor', 0.005)
        self.lr_critic = kwargs.get('lr_critic', 0.001)
        self.lr_scheduler = None
        self.criterion_critic = nn.MSELoss()
        self.compute_advantage_method = kwargs.get('compute_advantage_method', 'gae')
        # nn
        self.embedding_dim = kwargs.get('embedding_dim', 64)
        self.dropout_prob = kwargs.get('dropout_prob', 0.5)
        self.batch_norm = kwargs.get('batch_norm', False)
        # train
        self.batch_size = kwargs.get('batch_size', 128)
        self.use_negative_sample = kwargs.get('use_negative_sample', False)
        self.target_steps = kwargs.get('target_steps', 128)
        self.eval_interval = kwargs.get('eval_interval', 5)
        # eval;
        self.k_searching = kwargs.get('k_searching', 4)
        self.decode_strategy = kwargs.get('decode_strategy', 'sample')
        # tricks
        self.mask_actions = kwargs.get('mask_actions', True)
        self.maskable_policy = kwargs.get('maskable_policy', True)
        self.norm_reward = kwargs.get('norm_reward', False)
        if self.norm_reward:
            self.running_stats = RunningMeanStd(shape=1)
        self.norm_advantage = kwargs.get('norm_advantage', True)
        self.clip_grad = kwargs.get('clip_grad', True)
        self.clip_reward = kwargs.get('clip_reward', False)
        self.clip_reward = kwargs.get('max_reward', 1.0)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.)
        self.softmax_temp = 1.
        # log
        self.open_tb = kwargs.get('open_tb', True)
        self.log_dir = os.path.join(self.save_dir, 'log')
        self.model_dir = os.path.join(self.save_dir, 'model')
        self.writer = SummaryWriter(self.log_dir) if self.open_tb else None
        self.training_info = []
        self.buffer = RolloutBuffer()
        # save
        self.log_interval = kwargs.get('log_interval', 1)
        self.save_interval = kwargs.get('save_interval', 1)

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
        print(f'  distributed: {self.distributed_training}')
        print(f'     rl_gamma: {self.gamma}')
        print(f'     lr_actor: {self.lr_actor}')
        print(f'    lr_critic: {self.lr_critic}')
        print(f'   batch_size: {self.batch_size}')
        print(f'embedding_dim: {self.embedding_dim}')
        print(f'coef_ent_loss: {self.coef_entropy_loss}')
        print(f'     norm_adv: {self.norm_advantage}')
        print(f'  norm_reward: {self.norm_advantage}')
        print(f'    clip_grad: {self.clip_grad}')
        print(f'max_grad_norm: {self.max_grad_norm}')
        print(f'save_interval: {self.save_interval}')
        print(f'eval_interval: {self.eval_interval}')
        print(f'*' * 50)
        print()
        print(f'Logging training info at {os.path.join(self.log_dir, "training_info.csv")}')

    @abstractmethod
    def preprocess_obs(self, obs):
        return NotImplementedError

    def solve_with_baseline(self, instance, baseline='grc'):
        if self.baseline_solver is None:
            self.baseline_solver = GRCRankSolver()
        solution = self.baseline_solver.solve(instance)
        solution_info = self.counter.count_solution(instance['v_net'], solution)
        return solution_info

    def log(self, info, update_time, only_tb=False):
        if self.open_tb and hasattr(self, 'writer') and self.writer is not None:
            for key, value in info.items():
                self.writer.add_scalar(key, value, update_time)
        if only_tb:
            return
        log_fpath = os.path.join(self.log_dir, 'training_info.csv')
        write_head = not os.path.exists(log_fpath)
        with open(log_fpath, 'a+', newline='') as f:
            writer = csv.writer(f)
            if write_head:
                writer.writerow(['update_time'] + list(info.keys()))
            writer.writerow([update_time] + list(info.values()))
        if self.verbose >= 1:
            if update_time ==0 or update_time % 1000 == 0:
                info_key_str = ' & '.join([f'{k}' for k, v in info.items() if sum([s in k for s in ['loss', 'prob', 'return', 'penalty']])])
                print(f'             {update_time:06d} | ' + info_key_str)
            info_str = ' & '.join([f'{v:+3.4f}' for k, v in info.items() if sum([s in k for s in ['loss', 'prob', 'return', 'penalty']])])
            print(f'Update time: {update_time:06d} | ' + info_str)

    def get_action_prob_dist(self, observation):
        with torch.no_grad():
            action_logits = self.policy.act(observation)
        if 'action_mask' in observation and self.mask_actions:
            mask = observation['action_mask']
            candicate_action_logits = apply_mask_to_logit(action_logits, mask) 
        else:
            candicate_action_logits = action_logits
        action_prob_dist = F.softmax(candicate_action_logits / self.softmax_temp, dim=-1)
        return action_prob_dist, candicate_action_logits

    def select_action(self, observation, sample=True):
        with torch.no_grad():
            action_logits = self.policy.act(observation)
        if 'action_mask' in observation and self.mask_actions:
            mask = observation['action_mask']
            candicate_action_logits = apply_mask_to_logit(action_logits, mask) 
        else:
            candicate_action_logits = action_logits
        # if self.mask_actions and self.maskable_policy:
        #     candicate_action_probs = F.softmax(candicate_action_logits / self.softmax_temp, dim=-1)
        #     candicate_action_dist = Categorical(probs=candicate_action_probs)
        # else:
        #     candicate_action_probs = F.softmax(action_logits / self.softmax_temp, dim=-1)
        #     candicate_action_dist = Categorical(probs=candicate_action_probs)
        candicate_action_dist = Categorical(logits=candicate_action_logits / self.softmax_temp)
        raw_action_dist = Categorical(logits=action_logits / self.softmax_temp)

        if self.mask_actions and self.maskable_policy:
            action_dist_for_log_prob = candicate_action_dist
        else:
            action_dist_for_log_prob = raw_action_dist

        if sample:
            action = candicate_action_dist.sample()
        else:
            action = candicate_action_logits.argmax(-1)

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
            candicate_actions_logits = apply_mask_to_logit(actions_logits, masks)
        else:
            masks = None
            candicate_actions_logits = actions_logits

        candicate_actions_probs = F.softmax(candicate_actions_logits, dim=-1)

        dist = Categorical(actions_probs)
        candicate_dist = Categorical(candicate_actions_probs)
        policy_dist = candicate_dist if self.mask_actions and self.maskable_policy else dist

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
        unnecessary_attributes = ['policy', 'optimizer', 'lr_scheduler', 'searcher', 'writer']
        for attr_name in unnecessary_attributes:
            if hasattr(self, attr_name):
                temp_dict[attr_name] = getattr(self, attr_name)
                delattr(self, attr_name)
        sub_solver = copy.deepcopy(self)
        sub_solver.policy, _ = sub_solver.make_policy(sub_solver)
        for attr_name in temp_dict.keys():
            setattr(self, attr_name, temp_dict[attr_name])
        sub_solver.policy.load_state_dict(self.policy.state_dict())
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
        print(f'Save model to {checkpoint_fname}\n') if self.verbose >= 0 else None

    def load_model(self, checkpoint_path):
        print('Attempting to load the pretrained model')
        try:
            checkpoint = torch.load(checkpoint_path)
            if 'policy' not in checkpoint:
                self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
            else:
                self.policy.load_state_dict(checkpoint['policy'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Loaded pretrained model from {checkpoint_path}') if self.verbose >= 0 else None
        except Exception as e:
            print(f'Load failed from {checkpoint_path}\nInitilized with random parameters') if self.verbose >= 0 else None

    def train(self):
        """Set the mode to train"""
        self.policy.train()
        if hasattr(self, 'searcher'):
            delattr(self, 'searcher')

    def eval(self, decode_strategy=None, k=None):
        if decode_strategy is None: decode_strategy = self.decode_strategy
        if k is None: k = self.k_searching
        assert k >= 1, f'k should greater than 0. (k={k})'
        self.policy.eval()
        self.searcher = get_searcher(decode_strategy, 
                                    policy=self.policy, 
                                    preprocess_obs_func=self.preprocess_obs, 
                                    make_policy_func=self.make_policy,
                                    k=k, device=self.device,
                                    mask_actions=self.mask_actions, 
                                    maskable_policy=self.maskable_policy)

    def update_grad(self, loss):
        # update parameters
        if self.distributed_training:
            with self.lock:
                self.optimizer.zero_grad()
                self.shared_optimizer.zero_grad()
                loss.backward()
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) if self.clip_grad else None
                sync_gradients(self.shared_policy, self.policy)
                self.optimizer.step()
                self.shared_optimizer.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) if self.clip_grad else None
            self.optimizer.step()
        return grad_clipped

    def sync_parameters(self):
        assert self.distributed_training, 'distributed_training should be True'
        with self.lock:
            self.policy.load_state_dict(self.shared_policy.state_dict())

    def learn(self, env, num_epochs=1, **kwargs):
        self.start_time = time.time()
        if self.distributed_training:
            self.learn_distributedly(env, num_epochs)
        else:
            print(f'Start to learn singly')
            self.learn_singly(env, num_epochs)
        print(f'Start to validate')
        self.save_model(f'model.pkl')
        # self.validate(env)
        self.end_time = time.time()
        print(f'\nTotal training time: {(self.end_time - self.start_time) / 3600:4.6f} h')

    def learn_distributedly(self, env, num_epochs, **kwargs):
        assert self.distributed_training, 'distributed_training should be True'
        assert num_epochs % self.num_workers == 0, 'num_epochs should be divisible by num_workers'
        job_list = []
        mp.set_start_method('spawn')
        lock = mp.Lock()
        worker_num_epochs = int(num_epochs // self.num_workers)
        worker_save_interval = int(np.ceil(self.save_interval / self.num_workers))
        worker_eval_interval = int(np.ceil(self.eval_interval / self.num_workers))
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
    
    def __init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs):
        super(PGSolver, self).__init__(controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)

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
        self.log(info, self.update_time, only_tb=False)

        self.buffer.clear()
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        self.update_time += 1

        if self.distributed_training: self.sync_parameters()
        return loss.detach()


class A2CSolver(RLSolver):

    def __init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs):
        super(A2CSolver, self).__init__(controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        self.repeat_times = 1

    def update(self, ):
        observations = self.preprocess_obs(self.buffer.observations, self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
        advantages = returns - values.detach()
        if self.norm_advantage:
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
        self.log(info, self.update_time, only_tb=False)

        self.buffer.clear()
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        self.update_time += 1

        if self.distributed_training: self.sync_parameters()
        return loss.detach()


# class A3CSolver(A2CSolver):

#     def __init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs):
#         super(A3CSolver, self).__init__(controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
#         self.repeat_times = 1


class PPOSolver(RLSolver):

    def __init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs):
        super(PPOSolver, self).__init__(controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
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
        if self.norm_reward:
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
            if self.norm_advantage and values.numel() != 0:
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
        
            if self.open_tb and self.update_time % self.log_interval == 0:
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
                only_tb = not (i == sample_times-1)
                self.log(info, self.update_time, only_tb=only_tb)

            self.update_time += 1

        # print(f'loss: {loss.detach():+2.4f} = {actor_loss.detach():+2.4f} & {critic_loss:+2.4f} & {entropy_loss:+2.4f} & {mask_loss:+2.4f}, ' +
        #         f'action log_prob: {action_logprobs.mean():+2.4f} (old: {batch_old_action_logprobs.detach().mean():+2.4f}), ' +
        #         f'mean reward: {returns.detach().mean():2.4f}', file=self.fwriter) if self.verbose >= 0 else None
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()

        if self.distributed_training: self.sync_parameters()
        return loss.detach()


class A3CSolver(PPOSolver):

    def __init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs):
        super(A3CSolver, self).__init__(controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)



class ARPPOSolver(RLSolver):

    def __init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs):
        super(ARPPOSolver, self).__init__(controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
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
            if self.norm_advantage and values.numel() != 0:
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
    
            if self.open_tb and self.update_time % self.log_interval == 0:
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
                only_tb = not (i == sample_times-1)
                self.log(info, self.update_time, only_tb=only_tb)

            self.update_time += 1

        # print(f'loss: {loss.detach():+2.4f} = {actor_loss.detach():+2.4f} & {critic_loss:+2.4f} & {entropy_loss:+2.4f} & {mask_loss:+2.4f}, ' +
        #         f'action log_prob: {action_logprobs.mean():+2.4f} (old: {batch_old_action_logprobs.detach().mean():+2.4f}), ' +
        #         f'mean reward: {returns.detach().mean():2.4f}', file=self.fwriter) if self.verbose >= 0 else None
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        self.buffer.clear()

        if self.distributed_training: self.sync_parameters()
        return loss.detach()


class DDPGSolver(RLSolver):

    def __init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs):
        super(DDPGSolver, self).__init__(controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
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
        if self.norm_reward:
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
            if self.norm_advantage and values.numel() != 0:
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
        
            if self.open_tb and self.update_time % self.log_interval == 0:
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
                only_tb = not (i == sample_times-1)
                self.log(info, self.update_time, only_tb=only_tb)

            self.update_time += 1

        # print(f'loss: {loss.detach():+2.4f} = {actor_loss.detach():+2.4f} & {critic_loss:+2.4f} & {entropy_loss:+2.4f} & {mask_loss:+2.4f}, ' +
        #         f'action log_prob: {action_logprobs.mean():+2.4f} (old: {batch_old_action_logprobs.detach().mean():+2.4f}), ' +
        #         f'mean reward: {returns.detach().mean():2.4f}', file=self.fwriter) if self.verbose >= 0 else None
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()

        if self.distributed_training: self.sync_parameters()
        return loss.detach()
