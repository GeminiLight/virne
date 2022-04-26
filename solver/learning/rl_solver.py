import os
import csv
import time
import tqdm
import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from abc import abstractmethod

from base.recorder import Solution
from solver import Solver
from .buffer import ReplayBuffer
from .utils import apply_mask_to_logit, get_observations_sample


class RLSolver(Solver):

    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        super(RLSolver, self).__init__(name, reusable=reusable, verbose=verbose, **kwargs)
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
        self.coef_critic_loss = kwargs.get('coef_critic_loss', 0.5)
        self.coef_entropy_loss = kwargs.get('coef_entropy_loss', 0.01)
        self.coef_mask_loss = kwargs.get('coef_mask_loss', 0.01)
        self.weight_decay = kwargs.get('weight_decay', 0.00001)
        self.lr_actor = kwargs.get('lr_actor', 0.005)
        self.lr_critic = kwargs.get('lr_critic', 0.001)
        self.lr_scheduler = None
        self.criterion_critic = nn.MSELoss()
        # nn
        self.embedding_dim = kwargs.get('embedding_dim', 64)
        self.dropout_prob = kwargs.get('dropout_prob', 0.5)
        self.batch_norm = kwargs.get('batch_norm', False)
        # train
        self.batch_size = kwargs.get('batch_size', 128)
        self.use_negative_sample = kwargs.get('use_negative_sample', False)
        self.target_steps = kwargs.get('target_steps', 128)
        self.eval_interval = kwargs.get('eval_interval', 10)
        # tricks
        self.maskable_policy = kwargs.get('maskable_policy', True)
        self.norm_advantage = kwargs.get('norm_advantage', True)
        self.clip_grad = kwargs.get('clip_grad', True)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.)
        # log
        self.open_tb = kwargs.get('open_tb', True)
        self.log_dir = os.path.join(self.save_dir, 'log')
        self.model_dir = os.path.join(self.save_dir, 'model')
        self.writer = SummaryWriter(self.log_dir) if self.open_tb else None
        self.training_info = []
        self.buffer = ReplayBuffer()
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
            print(f'*' * 50)
            print(f'Key parameters of RL training are as following: ')
            print(f'*' * 50)
            print(f'       device: {self.device_name}')
            print(f'     parallel: {self.allow_parallel}')
            print(f'     rl_gamma: {self.gamma}')
            print(f'     lr_actor: {self.lr_actor}')
            print(f'    lr_critic: {self.lr_critic}')
            print(f'   batch_size: {self.batch_size}')
            print(f'coef_ent_loss: {self.coef_entropy_loss}')
            print(f'     norm_adv: {self.norm_advantage}')
            print(f'     norm_adv: {self.norm_advantage}')
            print(f'    clip_grad: {self.clip_grad}')
            print(f'max_grad_norm: {self.max_grad_norm}')
            print(f'*' * 50)
            print()
            print(f'Logging training info at {os.path.join(self.log_dir, "training_info.csv")}')

    @abstractmethod
    def preprocess_obs(self, obs):
        return NotImplementedError

    @abstractmethod
    def preprocess_batch_obs(self, obs_batch):
        return NotImplementedError

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
            info_str = ' & '.join([f'{v:+3.4f}' for k, v in info.items() if sum([s in k for s in ['loss', 'prob', 'return']])])
            print(f'Update time: {update_time:06d} | ' + info_str)

    def select_action(self, observation, mask=None, sample=True):
        observation = self.preprocess_obs(observation)
        with torch.no_grad():
            action_logits = self.policy.act(observation)

        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        candicate_action_logits = apply_mask_to_logit(action_logits, mask) if mask is not None else action_logits

        candicate_dist = Categorical(logits=candicate_action_logits)
        if sample:
            action = candicate_dist.sample()
        else:
            action = candicate_action_logits.argmax(-1)

        policy_dist = candicate_dist if self.maskable_policy else dist
        action_logprob = policy_dist.log_prob(action)

        return action.item(), action_logprob

    def evaluate_actions(self, old_observations, old_actions, masks=None, return_others=False):
        actions_logits = self.policy.act(old_observations)
        actions_probs = F.softmax(actions_logits, dim=-1)

        candicate_actions_logits = apply_mask_to_logit(actions_logits, masks) if masks is not None else actions_logits

        candicate_actions_probs = F.softmax(candicate_actions_logits, dim=-1)

        dist = Categorical(actions_probs)
        candicate_dist = Categorical(candicate_actions_probs)

        policy_dist = candicate_dist if self.maskable_policy else dist

        action_logprobs = policy_dist.log_prob(old_actions)
        dist_entropy = policy_dist.entropy()

        values = self.policy.evaluate(old_observations).squeeze(-1)

        if return_others:
            actions_probs
            mask_actions_probs = actions_probs * (~masks.bool())
            other = {
                'mask_actions_probs': mask_actions_probs.sum(-1)
            }
            return values, action_logprobs, dist_entropy, other

        return values, action_logprobs, dist_entropy

    def estimate_obs(self, observation):
        return self.policy.evaluate(observation).squeeze(-1)

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
        try:
            checkpoint = torch.load(checkpoint_path)
            if 'policy' not in checkpoint:
                self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
            else:
                self.policy.load_state_dict(checkpoint['policy'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Loaded pretrained model from {checkpoint_path}.') if self.verbose >= 0 else None
        except:
            print(f'Load failed from {checkpoint_path}\nInitilized with random parameters') if self.verbose >= 0 else None

    def solve(self, instance):
        vn, pn = instance['vn'], instance['pn']
        sub_env = self.SubEnv(pn, vn)
        obs = sub_env.get_observation()
        for vnf_id in list(vn.nodes):
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            action, action_logprob = self.select_action(obs, mask=mask, sample=False)
            obs, reward, done, info = sub_env.step(action)
            
            if done:
                return sub_env.curr_solution

    def validate(self, env, checkpoint_path=None):
        print(f"\n{'-' * 20}  Validate  {'-' * 20}\n") if self.verbose >= 0 else None
        if checkpoint_path is not None: self.load_model(checkpoint_path)

        pbar = tqdm.tqdm(desc=f'Validate {self.name}', total=env.num_vns) if self.verbose <= 1 else None
        
        instance = env.reset()
        while True:
            solution = self.solve(instance)
            next_instance, _, done, info = env.step(solution)

            if pbar is not None: 
                pbar.update(1)
                pbar.set_postfix({
                    'ac': f'{info["success_count"] / info["vn_count"]:1.2f}',
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


    def learn(self, env, num_epochs=1, start_epoch=0, save_timestep=1000, config=None, **kwargs):
        # main env
        self.start_time = time.time()
        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            print(f'Training Epoch: {epoch_id}')
            instance = env.reset()
            vn, pn = instance['vn'], instance['pn']
            success_count = 0
            epoch_logprobs = []
            for i in range(2000):
                revenue2cost_list = []
                ### --- sub env --- ###
                sub_buffer = ReplayBuffer()
                sub_env = self.SubEnv(pn, vn)
                sub_obs = sub_env.get_observation()
                for vnf_id in list(vn.nodes):
                    mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
                    action, action_logprob = self.select_action(sub_obs, mask=mask, sample=True)
                    value = self.estimate_obs(self.preprocess_obs(sub_obs))
                    next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action)

                    sub_buffer.add(sub_obs, action, sub_reward, sub_done, action_logprob, value=value)
                    sub_buffer.action_masks.append(mask)

                    if sub_done:
                        break

                    sub_obs = next_sub_obs
                    
                solution = sub_env.curr_solution
                if sub_env.curr_solution['result'] or self.use_negative_sample:  #  or True
                    revenue2cost_list.append(sub_reward)
                    if sub_env.curr_solution['result']: success_count = success_count + 1 
                    # sub_logprob = torch.cat(sub_logprob_list, dim=0).mean().unsqueeze(dim=0)
                    sub_buffer.compute_mc_returns(gamma=self.gamma)
                    self.buffer.merge(sub_buffer)
                    epoch_logprobs += sub_buffer.logprobs
                    self.time_step += 1
                else:
                    pass

                # update parameters
                if self.buffer.size() >= self.target_steps:
                    loss = self.update()
                    # print(f'loss: {loss.item():+2.4f}, mean r2c: {np.mean(revenue2cost_list):+2.4f}')

                ### --- sub env --- ###
                instance, reward, done, info = env.step(solution)

                # instance = env.reset()
                # vn, pn = instance['vn'], instance['pn']
                # epoch finished
                if not done:
                    vn, pn = instance['vn'], instance['pn']
                else:
                    break
            summary_info = env.summary_records()
            epoch_logprobs_tensor = torch.cat(epoch_logprobs, dim=0)
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["total_r2c"]:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}')
            self.save_model(f'model-{epoch_id}.pkl')
            # validate
            if (epoch_id + 1) != (start_epoch + num_epochs) and (epoch_id + 1) % self.eval_interval == 0:
                self.validate(env)

        self.end_time = time.time()
        print(f'\nTotal training time: {(self.end_time - self.start_time) / 3600:4.6f}')


    def online_learn(self, env, num_epochs=1, start_epoch=0, batch_size=32, save_timestep=1000, config=None):
        # main env
        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            obs = env.reset()
            success_count = 0
            for i in range(2000):

                mask = np.expand_dims(env.generate_action_mask(), axis=0)
                action, action_logprob = self.select_action(obs, mask=mask, sample=True)
                value = self.estimate_obs(self.preprocess_obs(obs))
                next_obs, reward, done, info = env.step(action)

                self.buffer.add(obs, action, reward, done, action_logprob, value=value)
                self.buffer.action_masks.append(mask)

                if done:
                    self.buffer.clear()
                    break

                obs = next_obs
                
                last_value = self.estimate_obs(self.preprocess_obs(next_obs))
                self.buffer.compute_returns_and_advantages(self, last_value, gamma=self.rl_gamma, gae_lambda=self.gae_lambda)
                
                self.time_step += 1

            # update parameters
            if self.buffer.size() >= batch_size:
                loss = self.update()
                # print(f'loss: {loss.item():+2.4f}, mean r2c: {np.mean(revenue2cost_list):+2.4f}')

            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["total_r2c"]:1.4f}')
            self.save_model(f'model-{epoch_id}.pkl')


class PGSolver(RLSolver):
    
    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        super(PGSolver, self).__init__(name, reusable, verbose, **kwargs)

    def update(self, ):
        observations = self.preprocess_batch_obs(self.buffer.observations)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
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


class A2CSolver(RLSolver):

    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        super(A2CSolver, self).__init__(name, reusable, verbose, **kwargs)
        self.repeat_times = 1

    def update(self, ):
        batch_logprobs = torch.cat(self.buffer.logprobs, dim=0).to(self.device)
        # batch_values = torch.cat(self.buffer.values, dim=0).to(self.device)
        observations = self.preprocess_batch_obs(self.buffer.observations)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device) if len(self.buffer.action_masks) != 0 else None
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

    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        super(PPOSolver, self).__init__(name, reusable, verbose, **kwargs)
        self.repeat_times = kwargs.get('repeat_times', 10)
        self.gae_lambda = kwargs.get('gae_lambda', 0.98)
        self.eps_clip = kwargs.get('eps_clip', 0.2)

    def update(self, ):
        assert self.buffer.size() >= self.batch_size

        batch_observations = self.preprocess_batch_obs(self.buffer.observations)
        batch_actions = torch.LongTensor(self.buffer.actions).to(self.device)
        batch_old_action_logprobs = torch.cat(self.buffer.logprobs, dim=0).to(self.device).detach()
        batch_rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        batch_returns = torch.FloatTensor(self.buffer.returns).to(self.device)

        batch_masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device) if len(self.buffer.action_masks) != 0 else None

        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            observations  = get_observations_sample(batch_observations, sample_indices)
            actions, returns = batch_actions[sample_indices], batch_returns[sample_indices]
            old_action_logprobs = batch_old_action_logprobs[sample_indices]
            masks = batch_masks[sample_indices]
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
            
            # calculate advantage
            advantage = returns - values.detach()
            if self.norm_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = self.criterion_critic(returns, values)
            entropy_loss = dist_entropy.mean()
            mask_loss = other['mask_actions_probs'].mean()
            loss = actor_loss + self.coef_critic_loss * critic_loss - self.coef_entropy_loss * entropy_loss + self.coef_mask_loss * mask_loss
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
                    'value/old_action_logprobs': old_action_logprobs.mean().cpu().numpy(),
                    'value/values': values.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
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
