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
from .utils import apply_mask_to_logit, get_observations_sample, RunningMeanStd
from virne.utils import test_running_time


class OnlineAgent(object):
    """Training and Inference Methods for Admission Control Agent"""
    def __init__(self) -> None:
        pass

    def solve(self, instance):
        instance = self.preprocess_obs(instance, self.device)
        action, action_logprob = self.select_action(instance, mask=None, sample=False)
        return action[0]

    def learn(self, env, num_epochs=1, start_epoch=0, save_timestep=1000, config=None, **kwargs):
        # main env
        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            obs = env.reset()
            success_count = 0
            for i in range(env.num_v_nets):
                tensor_obs = self.preprocess_obs(obs, self.device)
                action, action_logprob = self.select_action(tensor_obs, mask=None, sample=True)
                with torch.no_grad():
                    value = self.estimate_obs(tensor_obs)
                next_obs, reward, done, info = env.step(action[0])
                print(f'reward: {reward:2.2f}, value: {value.item():2.2f}, action_prob: {action_logprob.exp().item():2.2f}')
                self.buffer.add(obs, action, reward, done, action_logprob, value=value)
                obs = next_obs
                self.time_step += 1
                # update parameters
                if done and self.buffer.size() >= self.batch_size:  # done and 
                    with torch.no_grad():
                        last_value = float(self.estimate_obs(self.preprocess_obs(next_obs, self.device)).detach()[0]) if not done else 0.
                    # if self.norm_reward:
                        # self.running_stats.update(self.buffer.rewards)
                        # self.buffer.rewards = ((np.array(self.buffer.rewards) - self.running_stats.mean) / (np.sqrt(self.running_stats.var + 1e-9))).tolist()
                    self.buffer.compute_returns_and_advantages(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda, method=self.compute_return_method)
                    loss = self.update()
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["total_r2c"]:1.4f}, {self.running_stats.mean}-{np.sqrt(self.running_stats.var)}')
            if (epoch_id + 1) != (start_epoch + num_epochs) and (epoch_id + 1) % self.eval_interval == 0:
                self.validate(env)
            if (epoch_id + 1) != (start_epoch + num_epochs) and (epoch_id + 1) % self.save_interval == 0:
                self.save_model(f'model-{epoch_id}.pkl')

    def validate(self, env, checkpoint_path=None):
        print(f"\n{'-' * 20}  Validate  {'-' * 20}\n") if self.verbose >= 0 else None
        if checkpoint_path is not None: self.load_model(checkpoint_path)

        pbar = tqdm.tqdm(desc=f'Validate', total=env.num_v_nets) if self.verbose <= 1 else None
        
        instance = env.reset(0)
        while True:
            solution = self.solve(instance)
            next_instance, _, done, info = env.step(solution)

            if pbar is not None: 
                pbar.update(1)
                pbar.set_postfix({
                    'ac': f'{info["success_count"] / info["v_net_count"]:1.2f}',
                    'r2c': f'{info["total_r2c"]:1.2f}',
                    'inservice': f'{info["inservice_count"]:05d}',
                })

            if done:
                break
            instance = next_instance

        if pbar is not None: pbar.close()
        print(f"\n{'-' * 20}     Done    {'-' * 20}\n") if self.verbose >= 0 else None


class InstanceAgent(object):
    """Training and Inference Methods for Resource Allocation Agent"""
    def __init__(self) -> None:
        pass

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.SubEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        solution = self.searcher.find_solution(sub_env)
        return solution

    def validate(self, env, checkpoint_path=None):
        print(f"\n{'-' * 20}  Validate  {'-' * 20}\n") if self.verbose >= 0 else None
        if checkpoint_path is not None: self.load_model(checkpoint_path)

        pbar = tqdm.tqdm(desc=f'Validate', total=env.num_v_nets) if self.verbose <= 1 else None
        
        self.eval()
        instance = env.reset(0)
        while True:
            solution = self.solve(instance)
            next_instance, _, done, info = env.step(solution)

            if pbar is not None: 
                pbar.update(1)
                pbar.set_postfix({
                    'ac': f'{info["success_count"] / info["v_net_count"]:1.2f}',
                    'r2c': f'{info["total_r2c"]:1.2f}',
                    'inservice': f'{info["inservice_count"]:05d}',
                })

            if done:
                break
            instance = next_instance

        if pbar is not None: pbar.close()
        summary_info = env.summary_records()
        if self.verbose == 0:
            pprint.pprint(summary_info)
        print(f"\n{'-' * 20}     Done    {'-' * 20}\n") if self.verbose >= 0 else None

    def get_baseline_solution_info(self, instance, use_baseline_solver=True):
        """
        Get the baseline solution info for the instance, including:
            - result: whether the baseline solution is feasible
            - v_net_r2c_ratio: the revenue to cost ratio of the baseline solution

        Args:
            instance (dict): the instance to be solved
            use_baseline_solver (bool, optional): whether to use the baseline solver. Defaults to True.

        Returns:
            dict: the baseline solution info
        """
        if not use_baseline_solver:
            return {
                'result': True,
                'v_net_r2c_ratio': 0
            }
        baseline_solution = self.baseline_solver.solve(instance)
        baseline_solution_info = self.counter.count_solution(instance['v_net'], baseline_solution)
        return baseline_solution_info

    def learn_with_instance(self, instance):
        # sub env for sub agent
        sub_buffer = RolloutBuffer()
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.SubEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        sub_obs = sub_env.get_observation()
        sub_done = False
        while not sub_done:
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            tensor_sub_obs = self.preprocess_obs(sub_obs, self.device)
            action, action_logprob = self.select_action(tensor_sub_obs, mask=mask, sample=True)
            value = self.estimate_obs(tensor_sub_obs) if hasattr(self.policy, 'evaluate') else None
            next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action[0])

            sub_buffer.add(sub_obs, action, sub_reward, sub_done, action_logprob, value=value)
            sub_buffer.action_masks.append(mask)

            if sub_done:
                break

            sub_obs = next_sub_obs

        solution = sub_env.solution
        last_value = self.estimate_obs(self.preprocess_obs(next_sub_obs, self.device)) if hasattr(self.policy, 'evaluate') else None
        return solution, sub_buffer, last_value

    def merge_instance_experience(self, instance, solution, sub_buffer, last_value):
        ### -- use_negative_sample -- ##
        if self.use_negative_sample:
            baseline_solution_info = self.get_baseline_solution_info(instance, self.use_baseline_solver)
            if baseline_solution_info['result'] or solution['result']:
                sub_buffer.compute_returns_and_advantages(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method)
                self.buffer.merge(sub_buffer)
                self.time_step += 1
            else:
                pass
        elif solution['result']:  #  or True
            sub_buffer.compute_mc_returns(gamma=self.gamma)
            self.buffer.merge(sub_buffer)
            self.time_step += 1
        else:
            pass
        return self.buffer

    def learn(self, env, num_epochs=1, start_epoch=0, save_timestep=1000, config=None, **kwargs):
        # main env
        self.start_time = time.time()
        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            print(f'Training Epoch: {epoch_id}')
            instance = env.reset()
            success_count = 0
            epoch_logprobs = []
            revenue2cost_list = []
            for i in range(env.num_v_nets):
                ### --- instance-level --- ###
                solution, sub_buffer, last_value = self.learn_with_instance(instance)
                epoch_logprobs += sub_buffer.logprobs
                self.merge_instance_experience(instance, solution, sub_buffer, last_value)

                if solution.is_feasible():
                    success_count += 1
                    revenue2cost_list.append(solution['v_net_r2c_ratio'])
                # update parameters
                if self.buffer.size() >= self.target_steps:
                    loss = self.update()
                    # print(f'loss: {loss.item():+2.4f}, mean r2c: {np.mean(revenue2cost_list):+2.4f}')

                instance, reward, done, info = env.step(solution)
                # instance = env.reset()
                # epoch finished
                if done:
                    break
                
            summary_info = env.summary_records()
            epoch_logprobs_tensor = np.concatenate(epoch_logprobs, axis=0)
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["total_r2c"]:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}')
            # save
            if (epoch_id + 1) != (start_epoch + num_epochs) and (epoch_id + 1) % self.save_interval == 0:
                self.save_model(f'model-{epoch_id}.pkl')
            # validate
            if (epoch_id + 1) != (start_epoch + num_epochs) and (epoch_id + 1) % self.eval_interval == 0:
                self.validate(env)

        self.end_time = time.time()
        print(f'\nTotal training time: {(self.end_time - self.start_time) / 3600:4.6f} h')


class RLSolver(Solver):
    """General Reinforcement Learning Solve"""
    def __init__(self, controller, recorder, counter, **kwargs):
        super(RLSolver, self).__init__(controller, recorder, counter, **kwargs)
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
        self.allow_parallel = kwargs.get('allow_parallel', False)
        # rl
        self.gamma = kwargs.get('rl_gamma', 0.95)
        self.gae_lambda = kwargs.get('gae_lambda', 0.98)
        self.coef_critic_loss = kwargs.get('coef_critic_loss', 0.5)
        self.coef_entropy_loss = kwargs.get('coef_entropy_loss', 0.01)
        self.coef_mask_loss = kwargs.get('coef_mask_loss', 0.01)
        self.weight_decay = kwargs.get('weight_decay', 0.00001)
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

    def show_config(self, ):
        print(f'*' * 50)
        print(f'Key parameters of RL training are as following: ')
        print(f'*' * 50)
        print(f'       device: {self.device_name}')
        print(f'     parallel: {self.allow_parallel}')
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
        if self.open_tb:
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

    def get_action_prob_dist(self, observation, mask=None):
        with torch.no_grad():
            action_logits = self.policy.act(observation)
        if mask is not None and self.mask_actions:
            candicate_action_logits = apply_mask_to_logit(action_logits, mask) 
        else:
            candicate_action_logits = action_logits
        action_prob_dist = F.softmax(candicate_action_logits / self.softmax_temp, dim=-1)
        return action_prob_dist, candicate_action_logits

    def select_action(self, observation, mask=None, sample=True):
        with torch.no_grad():
            action_logits = self.policy.act(observation)

        if mask is not None and self.mask_actions:
            candicate_action_logits = apply_mask_to_logit(action_logits, mask) 
        else:
            candicate_action_logits = action_logits

        if self.mask_actions and self.maskable_policy:
            candicate_action_probs = F.softmax(candicate_action_logits / self.softmax_temp, dim=-1)
            candicate_action_dist = Categorical(probs=candicate_action_probs)
        else:
            candicate_action_probs = F.softmax(action_logits / self.softmax_temp, dim=-1)
            candicate_action_dist = Categorical(probs=candicate_action_dist)

        if sample:
            action = candicate_action_dist.sample()
        else:
            action = candicate_action_logits.argmax(-1)

        action_logprob = candicate_action_dist.log_prob(action)
        action = action.reshape(-1, )
        # action = action.squeeze(-1).cpu()
        return action.cpu().detach().numpy(), action_logprob.cpu().detach().numpy()

    def evaluate_actions(self, old_observations, old_actions, masks=None, return_others=False):
        actions_logits = self.policy.act(old_observations)
        actions_probs = F.softmax(actions_logits / self.softmax_temp, dim=-1)

        if masks is not None:
            candicate_actions_logits = apply_mask_to_logit(actions_logits, masks)
        else:
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

    def estimate_obs(self, observation):
        return self.policy.evaluate(observation).squeeze(-1).detach().cpu()

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
                                    k=k, device=self.device,
                                    mask_actions=self.mask_actions, 
                                    maskable_policy=self.maskable_policy)

    def purify(self):
        purified_agent = copy.deepcopy(self)
        unnecessary_attributes = ['optimizor']
        for attr_name in unnecessary_attributes:
            if hasattr(purified_agent, attr_name):
                delattr(purified_agent, attr_name)
        return purified_agent


class PGSolver(RLSolver):
    
    def __init__(self, controller, recorder, counter, **kwargs):
        super(PGSolver, self).__init__(controller, recorder, counter, **kwargs)

    def update(self, ):
        observations = self.preprocess_obs(self.buffer.observations, self.device)
        actions = torch.LongTensor(np.concatenate(self.buffer.actions, axis=0)).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device) if len(self.buffer.action_masks) != 0 else None
        _, action_logprobs, _, _ = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
        
        loss = - (action_logprobs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

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
        return loss


class PGWithBaselineSolver(RLSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(PGWithBaselineSolver, self).__init__(controller, recorder, counter, **kwargs)

    def update(self, ):
        observations = self.preprocess_obs(self.buffer.observations, self.device)
        actions = torch.LongTensor(np.concatenate(self.buffer.actions, axis=0)).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        if len(self.buffer.action_masks) != 0 and self.mask_actions:
            masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device)
        else:
            masks = None
        _, action_logprobs, _, _ = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
        
        loss = - (action_logprobs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

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
        return loss


class A2CSolver(RLSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(A2CSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.repeat_times = 1

    def update(self, ):
        observations = self.preprocess_obs(self.buffer.observations, self.device)
        actions = torch.LongTensor(np.concatenate(self.buffer.actions, axis=0)).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        if len(self.buffer.action_masks) != 0 and self.mask_actions:
            masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device)
        else:
            masks = None
        values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
        advantages = returns - values.detach()
        if self.norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = - (action_logprobs * advantages).mean()
        critic_loss = F.mse_loss(returns, values)
        entropy_loss = dist_entropy.mean()
        loss = actor_loss + self.coef_critic_loss * critic_loss + self.coef_entropy_loss * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

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
        return loss


class PPOSolver(RLSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(PPOSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.repeat_times = kwargs.get('repeat_times', 10)
        self.gae_lambda = kwargs.get('gae_lambda', 0.98)
        self.eps_clip = kwargs.get('eps_clip', 0.2)

    def update(self, ):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(np.concatenate(self.buffer.actions, axis=0)).to(self.device)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_returns = torch.FloatTensor(self.buffer.returns)

        if self.norm_reward:
            batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-9)

        if len(self.buffer.action_masks) != 0 and self.mask_actions:
            batch_masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0))
        else:
            batch_masks = None

        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            observations  = get_observations_sample(batch_observations, sample_indices, self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            masks = batch_masks[sample_indices].to(self.device) if batch_masks is not None else None
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
            
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
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
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
        return loss.detach()


class ARPPOSolver(RLSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(ARPPOSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.repeat_times = kwargs.get('repeat_times', 10)
        self.gae_lambda = kwargs.get('gae_lambda', 0.98)
        self.eps_clip = kwargs.get('eps_clip', 0.2)

    def update(self, ):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(np.concatenate(self.buffer.actions, axis=0)).to(self.device)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        mean_batch_rewards = batch_rewards.mean()

        batch_returns = torch.FloatTensor(self.buffer.returns)

        if len(self.buffer.action_masks) != 0 and self.mask_actions:
            batch_masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0))
        else:
            batch_masks = None

        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            observations = get_observations_sample(batch_observations, sample_indices, device=self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            masks = batch_masks[sample_indices].to(self.device) if batch_masks is not None else None
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
            
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
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
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
        return loss.detach()
