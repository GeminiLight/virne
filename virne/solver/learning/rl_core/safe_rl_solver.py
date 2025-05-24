# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pool
from torch.distributions import Categorical
from torch.nn.functional import softplus

from .rl_solver import *
from .buffer import RolloutBuffer, RolloutBufferWithCost
from ..utils import apply_mask_to_logit, get_observations_sample


class SafeRLSolver(RLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        self.penalty_params = torch.tensor(1.0, requires_grad=True).float()
        self.lr_penalty_params = kwargs.get('lr_penalty_params', 1e-3)
        self.criterion_cost_critic = nn.MSELoss()
        self.cost_budget = kwargs.get('cost_budget', 0.)
        self.lr_cost_critic = kwargs.get('lr_cost_critic', 1e-3)
        self.coef_cost_critic_loss = kwargs.get('coef_cost_critic_loss', 1e-2)
        print(f'Cost Budget: {self.cost_budget}')
        super(SafeRLSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.penalty_params.to(self.device)
        self.buffer = RolloutBufferWithCost()
        self.compute_cost_method = kwargs.get('compute_cost_method', 'cumulative')
        self.cost_gamma = kwargs.get('cost_gamma', 1.)

    def save_model(self, checkpoint_fname):
        checkpoint_fname = os.path.join(self.model_dir, checkpoint_fname)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'penalty_params': self.penalty_params
        }, checkpoint_fname)
        self.logger.critical(f'Save model to {checkpoint_fname}\n')

    def load_model(self, checkpoint_fname):
        super().load_model(checkpoint_fname)
        checkpoint = torch.load(checkpoint_fname)
        if 'penalty_params' in checkpoint:
            self.penalty_params = checkpoint['penalty_params']

    def estimate_cost(self, observations):
        with torch.no_grad():
            cost_values = self.policy.evaluate_cost(observations).squeeze(-1)
        return cost_values

    def estimate_cost_with_grad(self, observations):
        cost_values = self.policy.evaluate_cost(observations).squeeze(-1)
        return cost_values


class FixedPenaltyPPOSolver(PPOSolver, SafeRLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(FixedPenaltyPPOSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.penalty_params.requires_grad = False

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size

        device = torch.device('cpu')
        batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(self.buffer.actions)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_costs = torch.FloatTensor(self.buffer.costs)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns)
        batch_returns = torch.FloatTensor(self.buffer.returns)
        # update the policy params repeatly
        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            cost_returns = batch_cost_returns[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            # calculate advantage
            advantages = returns - cost_returns * self.penalty_params - values.detach()
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            
            # calculate loss
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = self.criterion_critic(returns, values)
            entropy_loss = dist_entropy.mean()

            mask_loss = other.get('mask_actions_probs', 0)
            prediction_loss = other.get('prediction_loss', 0)

            loss = actor_loss \
                    + self.coef_critic_loss * critic_loss \
                    - self.coef_entropy_loss * entropy_loss \
                    + self.coef_mask_loss * mask_loss \
                    + prediction_loss
            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.rl.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
            self.optimizer.step()
    
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
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'cost_value/cost_return': cost_returns.mean().cpu().numpy(),
                    'cost_value/cost': batch_costs.mean().cpu().numpy(),
                    'penalty_params': self.penalty_params.detach().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                self.logger.log(data=info, step=self.update_time)

            self.update_time += 1

        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()


class LagrangianPPOSolver(PPOSolver, SafeRLSolver):

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(self.buffer.actions)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_costs = torch.FloatTensor(self.buffer.costs)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns)
        batch_returns = torch.FloatTensor(self.buffer.returns)
        # only optimize the penalty param once
        penalty_loss = - self.penalty_params * (avg_cost - self.cost_budget)
        self.optimizer.zero_grad()
        penalty_loss.backward(retain_graph=True)
        self.optimizer.step()
        cur_penalty = softplus(self.penalty_params).item()
        self.logger.info('loss: ', penalty_loss.item(), 'penalty_params: ', self.penalty_params, 'avg_cost: ', avg_cost)
        # update the policy params repeatly
        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            cost_returns = batch_cost_returns[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices]
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            cost_values = self.estimate_cost_with_grad(observations)
            
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = cost_returns - cost_values.detach()
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            
            # calculate loss
            actor_loss = - torch.min(surr1, surr2).mean() + cur_penalty * (ratio * cost_advantages).mean()
            critic_loss = self.criterion_critic(returns, values)
            cost_critic_loss = self.criterion_cost_critic(cost_returns, cost_values)
            entropy_loss = dist_entropy.mean()

            mask_loss = other.get('mask_actions_probs', 0)
            prediction_loss = other.get('prediction_loss', 0)

            loss = actor_loss \
                    + self.coef_critic_loss * critic_loss \
                    + self.coef_cost_critic_loss * cost_critic_loss \
                    - self.coef_entropy_loss * entropy_loss \
                    + self.coef_mask_loss * mask_loss \
                    + prediction_loss
            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.rl.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
            self.optimizer.step()
    
            if self.update_time % self.config.training.log_interval == 0:
                info = {
                    'lr': self.optimizer.defaults['lr'],
                    'loss/loss': loss.detach().cpu().numpy(),
                    'loss/actor_loss': actor_loss.detach().cpu().numpy(),
                    'loss/critic_loss': critic_loss.detach().cpu().numpy(),
                    'loss/cost_critic_loss': cost_critic_loss.detach().cpu().numpy(),
                    'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
                    'loss/penalty_loss': penalty_loss.detach().cpu().numpy(),
                    'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
                    'value/old_action_logprob': old_action_logprobs.mean().cpu().numpy(),
                    'value/value': values.detach().mean().cpu().numpy(),
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'cost_value/cost_value': cost_values.detach().mean().cpu().numpy(),
                    'cost_value/cost_advantages': cost_advantages.detach().mean().cpu().numpy(),
                    'cost_value/cost_return': cost_returns.mean().cpu().numpy(),
                    'cost_value/cost': batch_costs.mean().cpu().numpy(),
                    'penalty_params': self.penalty_params.detach().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                # only_tb = not (i == sample_times-1)
                self.logger.log(data=info, step=self.update_time)

            self.update_time += 1

        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()


class NeuralLagrangianPPOSolver(PPOSolver, SafeRLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(NeuralLagrangianPPOSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)

    def calc_penalty_params(self, observations):
        cost_values = self.policy.evaluate_lambda(observations).squeeze(-1)
        return F.softplus(cost_values)

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        # batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(self.buffer.actions)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_costs = torch.FloatTensor(self.buffer.costs)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns)
        batch_returns = torch.FloatTensor(self.buffer.returns)
        batch_cost_violations = batch_cost_returns - self.cost_budget
        # update the policy params repeatly
        # sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        sample_times = self.repeat_times
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions, returns = batch_actions[sample_indices].to(self.device), batch_returns[sample_indices].to(self.device)
            cost_returns = batch_cost_returns[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            cost_values = self.estimate_cost_with_grad(observations)
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = (cost_returns - self.cost_budget) - (cost_values.detach() - self.cost_budget)
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            # calculate loss
            cur_penalty = self.calc_penalty_params(observations).detach()
            # cur_penalty = torch.ones(size=(1,), device=self.device) * 1000
            actor_loss = - torch.min(surr1, surr2).mean() + (cur_penalty * (ratio * cost_advantages)).mean()
            critic_loss = self.criterion_critic(returns, values)
            cost_critic_loss = self.criterion_cost_critic(cost_returns, cost_values)
            entropy_loss = dist_entropy.mean()
            mask_loss = other.get('mask_actions_probs', 0)

            if (self.update_time+1) % (self.repeat_times) == 0:
                # Compute lam_net loss
                # current_penalty = torch.ones(size=(1,), device=self.device) * 1000
                cur_penalty = self.calc_penalty_params(observations)
                cost_values = self.estimate_cost(observations).detach()
                penalty_loss = -(cur_penalty * (cost_values - self.cost_budget)).mean()
                # penalty_loss = torch.zeros(size=(1,), device=self.device).mean()
                self.logger.info(f'curr_penalty: {cur_penalty.mean().item():.4f}, cost_budget {self.cost_budget:.4f}, penalty_loss: {penalty_loss.mean().item():.4f},  cost_returns: {cost_returns.mean().item():.4f},  cost_values: {cost_values.mean().item():.4f}, cost_critic_loss: {cost_critic_loss.mean().item():.4f}')
            else:
                penalty_loss = torch.zeros(size=(1,), device=self.device).mean()

            loss = actor_loss \
                    + self.coef_critic_loss * critic_loss \
                    + self.coef_cost_critic_loss * cost_critic_loss \
                    - self.coef_entropy_loss * entropy_loss \
                    + self.coef_mask_loss * mask_loss \
                    + penalty_loss
            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.rl.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
            self.optimizer.step()
    
            if self.update_time % self.config.training.log_interval == 0:
                info = {
                    'lr': self.optimizer.defaults['lr'],
                    'loss/loss': loss.detach().cpu().numpy(),
                    'loss/actor_loss': actor_loss.detach().cpu().numpy(),
                    'loss/critic_loss': critic_loss.detach().cpu().numpy(),
                    'loss/cost_critic_loss': cost_critic_loss.detach().cpu().numpy(),
                    'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
                    'loss/penalty_loss': penalty_loss.detach().cpu().numpy(),
                    'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
                    'value/old_action_logprob': old_action_logprobs.mean().cpu().numpy(),
                    'value/value': values.detach().mean().cpu().numpy(),
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'cost_value/cost_value': cost_values.detach().mean().cpu().numpy(),
                    'cost_value/cost_advantage': cost_advantages.detach().mean().cpu().numpy(),
                    'cost_value/cost_budget': self.cost_budget,
                    'cost_value/cost_return': batch_cost_returns.mean().cpu().numpy(),
                    'cost_value/cost_violation': batch_cost_violations.mean().cpu().numpy(),
                    'cost_value/cost': batch_costs.mean().cpu().numpy(),
                    # 'penalty_params': cur_penalty.mean().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                self.logger.log(data=info, step=self.update_time)

            self.update_time += 1

        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()



class RobustNeuralLagrangianPPOSolver(PPOSolver, SafeRLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(RobustNeuralLagrangianPPOSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.baseline_solver = self.to_sub_solver()
        if self.if_use_baseline_solver == True:
            self.baseline_solver.policy.eval()
            self.baseline_solver.searcher = get_searcher('greedy', 
                                        policy=self.baseline_solver.policy, 
                                        preprocess_obs_func=self.preprocess_obs, 
                                        k=1, device=self.device,
                                        mask_actions=self.config.rl.mask_actions, 
                                        maskable_policy=self.config.rl.maskable_policy,
                                        make_policy_func=self.make_policy,)

    def calc_penalty_params(self, observations):
        cost_values = self.policy.evaluate_lambda(observations).squeeze(-1)
        return F.softplus(cost_values)

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        # batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(self.buffer.actions)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_costs = torch.FloatTensor(self.buffer.costs)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns)
        batch_returns = torch.FloatTensor(self.buffer.returns)
        batch_feasibiliy_flags = torch.FloatTensor(self.buffer.feasibility_flags)
        import pdb; pdb.set_trace()
        self.logger.info(f'feasibility_flags: {batch_feasibiliy_flags.mean().item():.4f}')
        batch_cost_violations = batch_cost_returns - self.cost_budget
        # update the policy params repeatly
        # sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        sample_times = self.repeat_times
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions, returns = batch_actions[sample_indices].to(self.device), batch_returns[sample_indices].to(self.device)
            cost_returns = batch_cost_returns[sample_indices].to(self.device)
            feasibility_flags = batch_feasibiliy_flags[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            cost_values = self.estimate_cost_with_grad(observations)
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = (cost_returns - self.cost_budget) - (cost_values.detach() - self.cost_budget)
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            # calculate loss
            cur_penalty = self.calc_penalty_params(observations).detach()
            # cur_penalty = torch.ones(size=(1,), device=self.device) * 1000
            reward_loss = (- torch.min(surr1, surr2)).mean()
            #   * feasibility_flags
            cost_loss = (cur_penalty * (ratio * cost_advantages)).mean()
            actor_loss = reward_loss + cost_loss
            critic_loss = self.criterion_critic(returns, values)
            cost_critic_loss = self.criterion_cost_critic(cost_returns, cost_values)
            entropy_loss = dist_entropy.mean()
            mask_loss = other.get('mask_actions_probs', 0)

            if (self.update_time+1) % (self.repeat_times) == 0:
                # Compute lam_net loss
                # current_penalty = torch.ones(size=(1,), device=self.device) * 1000
                cur_penalty = self.calc_penalty_params(observations)
                cost_values = self.estimate_cost(observations).detach()
                cur_penalty = cur_penalty * feasibility_flags
                penalty_loss = -(cur_penalty * (cost_values - self.cost_budget)).mean()
                
                # penalty_loss = torch.zeros(size=(1,), device=self.device).mean()
                self.logger.info(f'curr_penalty: {cur_penalty.mean().item():.4f}, cost_budget {self.cost_budget:.4f}, penalty_loss: {penalty_loss.mean().item():.4f},  cost_returns: {cost_returns.mean().item():.4f},  cost_values: {cost_values.mean().item():.4f}, cost_critic_loss: {cost_critic_loss.mean().item():.4f}')
            else:
                penalty_loss = torch.zeros(size=(1,), device=self.device).mean()

            loss = actor_loss \
                    + self.coef_critic_loss * critic_loss \
                    + self.coef_cost_critic_loss * cost_critic_loss \
                    - self.coef_entropy_loss * entropy_loss \
                    + self.coef_mask_loss * mask_loss \
                    + penalty_loss
            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.rl.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
            self.optimizer.step()
    
            if self.update_time % self.config.training.log_interval == 0:
                info = {
                    'lr': self.optimizer.defaults['lr'],
                    'loss/loss': loss.detach().cpu().numpy(),
                    'loss/actor_loss': actor_loss.detach().cpu().numpy(),
                    'loss/critic_loss': critic_loss.detach().cpu().numpy(),
                    'loss/cost_critic_loss': cost_critic_loss.detach().cpu().numpy(),
                    'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
                    'loss/penalty_loss': penalty_loss.detach().cpu().numpy(),
                    'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
                    'value/old_action_logprob': old_action_logprobs.mean().cpu().numpy(),
                    'value/value': values.detach().mean().cpu().numpy(),
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'cost_value/cost_value': cost_values.detach().mean().cpu().numpy(),
                    'cost_value/cost_advantage': cost_advantages.detach().mean().cpu().numpy(),
                    'cost_value/cost_budget': self.cost_budget,
                    'cost_value/cost_return': batch_cost_returns.mean().cpu().numpy(),
                    'cost_value/cost_violation': batch_cost_violations.mean().cpu().numpy(),
                    'cost_value/cost': batch_costs.mean().cpu().numpy(),
                    # 'penalty_params': cur_penalty.mean().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                self.logger.log(data=info, step=self.update_time)

            self.update_time += 1

            if self.if_use_baseline_solver and self.update_time % 100 == 0:
                self.baseline_solver.policy.load_state_dict(self.policy.state_dict())
                self.logger.info(f'Update time == {self.update_time},  baseline_solver updated')

        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()


class RewardCPOSolver(PPOSolver, SafeRLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(RewardCPOSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.baseline_solver = self.to_sub_solver()
        self.if_use_baseline_solver = True
        self.baseline_solver.policy.eval()
        self.baseline_solver.searcher = get_searcher('greedy', 
                                    policy=self.baseline_solver.policy, 
                                    preprocess_obs_func=self.preprocess_obs, 
                                    k=1, device=self.device,
                                    mask_actions=self.config.rl.mask_actions, 
                                    maskable_policy=self.config.rl.maskable_policy,
                                    make_policy_func=self.make_policy,)
        self.fixed_lambda = kwargs.get('srl_lambda', 1)

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        # batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(self.buffer.actions)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_costs = torch.FloatTensor(self.buffer.costs)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns)
        batch_cost_budgets = torch.FloatTensor(self.buffer.baseline_cost_returns)
        batch_cost_violations = batch_cost_returns - batch_cost_budgets

        batch_returns = torch.FloatTensor(self.buffer.returns)
        # update the policy params repeatly
        # sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        sample_times = self.repeat_times
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long().to(self.device)
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions, returns = batch_actions[sample_indices].to(self.device), batch_returns[sample_indices].to(self.device)
            cost_returns = batch_cost_returns[sample_indices].to(self.device)
            cost_budgets = batch_cost_budgets[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            cost_values = self.estimate_cost_with_grad(observations)
            
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = cost_returns - cost_values.detach()
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            
            # calculate loss
            cur_penalty = torch.ones(size=(1,), device=self.device) * self.fixed_lambda
            actor_loss = - torch.min(surr1, surr2).mean() + (cur_penalty * (ratio * cost_advantages)).mean()
            critic_loss = self.criterion_critic(returns, values)
            cost_critic_loss = self.criterion_cost_critic(cost_returns, cost_values)
            entropy_loss = dist_entropy.mean()

            mask_loss = other.get('mask_actions_probs', 0)
            prediction_loss = other.get('prediction_loss', 0)

            if (self.update_time+1) % (self.repeat_times) == 0:
                # Compute lam_net loss
                # current_penalty = self.calc_penalty_params(observations)
                current_penalty = torch.ones(size=(1,), device=self.device) * self.fixed_lambda
                cost_values = self.estimate_cost(observations).detach()
                cost_budgets = torch.where(cost_budgets > cost_returns, cost_returns, cost_budgets)
                # penalty_loss = -(current_penalty * (cost_values - cost_budget)).mean()
                penalty_loss = -(current_penalty * (cost_returns - cost_budgets)).mean()
                penalty_loss = torch.zeros(size=(1,), device=self.device).mean()
                # print(f'cost_budget: {cost_budgets.mean().it():.4f}, mean_returns: {returns.mean().item():.4f}, mean_cost_returns: {cost_returns.mean().item():.4f}, curr_penalty: {current_penalty.mean().item():.4f}, penalty_loss: {penalty_loss.mean().item():.4f},  cost_values: {cost_values.mean().item():.4f}')
            else:
                penalty_loss = torch.zeros(size=(1,), device=self.device).mean()

            loss = actor_loss \
                    + self.coef_critic_loss * critic_loss \
                    + self.coef_cost_critic_loss * cost_critic_loss \
                    - self.coef_entropy_loss * entropy_loss \
                    + self.coef_mask_loss * mask_loss \
                    + prediction_loss \
                    + penalty_loss
            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.rl.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
            self.optimizer.step()
    
            if self.update_time % self.config.training.log_interval == 0:
                info = {
                    'lr': self.optimizer.defaults['lr'],
                    'loss/loss': loss.detach().cpu().numpy(),
                    'loss/actor_loss': actor_loss.detach().cpu().numpy(),
                    'loss/critic_loss': critic_loss.detach().cpu().numpy(),
                    'loss/cost_critic_loss': cost_critic_loss.detach().cpu().numpy(),
                    'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
                    'loss/penalty_loss': penalty_loss.detach().cpu().numpy(),
                    'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
                    'value/old_action_logprob': old_action_logprobs.mean().cpu().numpy(),
                    'value/value': values.detach().mean().cpu().numpy(),
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'cost_value/cost_value': cost_values.detach().mean().cpu().numpy(),
                    'cost_value/cost_advantage': cost_advantages.detach().mean().cpu().numpy(),
                    'cost_value/cost_budget': batch_cost_budgets.mean().cpu().numpy(),
                    'cost_value/cost_return': batch_cost_returns.mean().cpu().numpy(),
                    'cost_value/cost_violation': batch_cost_violations.mean().cpu().numpy(),
                    'cost_value/cost': batch_costs.mean().cpu().numpy(),
                    'penalty_params': cur_penalty.mean().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                self.logger.log(data=info, step=self.update_time)

            self.update_time += 1
        
            if self.update_time % 100 == 0:
                self.baseline_solver.policy.load_state_dict(self.policy.state_dict())
                self.logger.info(f'Update time == {self.update_time},  baseline_solver updated')

        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()


class AdaptiveStateWiseSafePPOSolver(PPOSolver, SafeRLSolver):

    def __init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs):
        super(AdaptiveStateWiseSafePPOSolver, self).__init__(controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.srl_alpha = kwargs.get('srl_alpha', 1.)
        self.srl_beta = kwargs.get('srl_beta', 0.1)

    def calc_penalty_params(self, observations):
        lambda_values = self.policy.lambda_net(observations).squeeze(-1)
        return torch.clip(F.softplus(lambda_values, beta=1, threshold=20), max=10)

    def estimate_budgets(self, observations):
        cost_budgets = self.policy.budget_net(observations).squeeze(-1)
        return F.softplus(cost_budgets, beta=1, threshold=20)

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(np.concatenate(self.buffer.actions, axis=0))
        batch_old_action_logprobs = torch.cat(self.buffer.logprobs, dim=0)
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_costs = torch.FloatTensor(self.buffer.costs)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns)
        batch_costs = batch_cost_returns
        batch_returns = torch.FloatTensor(self.buffer.returns)
        # update the policy params repeatly
        # sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        sample_times = self.repeat_times
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            observations  = get_observations_sample(batch_observations, sample_indices, device=self.device)
            actions = batch_actions[sample_indices].to(self.device)
            returns = batch_returns[sample_indices].to(self.device)
            costs = batch_costs[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            cost_values = self.estimate_cost_with_grad(observations)
            
            # calculate advantage
            # cur_penalty =  self.calc_penalty_params(observations).detach()
            # safety_budgets =  self.estimate_budgets(observations).detach()
            # return_with_costs = returns - cur_penalty * (costs - safety_budgets)
            # return_with_cost_values = values - cur_penalty * cur_penalty * (cost_values - safety_budgets)
            # advantages = return_with_costs - return_with_cost_values.detach()
            # if self.config.rl.norm_advantage and values.numel() != 0:
            #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            # ratio = torch.exp(action_logprobs - old_action_logprobs)
            # surr1 = ratio * advantages
            # surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            
            # Official
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = costs - cost_values.detach()
            if self.config.rl.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            
            # calculate loss
            cur_penalty =  self.calc_penalty_params(observations).detach()
            actor_loss = - torch.min(surr1, surr2).mean() + (cur_penalty * (ratio * cost_advantages)).mean()
            critic_loss = self.criterion_critic(returns, values)
            cost_critic_loss = self.criterion_cost_critic(costs, cost_values)
            entropy_loss = dist_entropy.mean()

            mask_loss = other.get('mask_actions_probs', 0)
            prediction_loss = other.get('prediction_loss', 0)

            if (self.update_time+1) % (self.repeat_times) == 0:
                # Compute lam_net loss
                values = self.estimate_value(observations)
                current_penalty = self.calc_penalty_params(observations)
                current_penalty = torch.clip(current_penalty, max=50)
                cost_values = self.estimate_cost(observations).detach()
                # current objective
                curr_objective =  returns - current_penalty * costs
                # print(curr_objective.mean())

                # print((sm_objective - curr_objective).mean())

                cost_budgets_scaler = self.estimate_budgets(observations)
                cost_budgets_scaler = torch.zeros_like(cost_values, device=self.device) + 1
                task_difficuties = torch.tanh(cost_values.detach() / values.detach())
                loss_func = nn.MSELoss()
                budget_loss = loss_func(cost_budgets_scaler, torch.zeros_like(cost_budgets_scaler, device=self.device))
                # lambda_loss = loss_func(current_penalty, torch.zeros_like(current_penalty, device=self.device))
                # cost_budgets =  cost_budgets_scaler * cost_values.detach()
                cost_budgets = cost_budgets_scaler
                penalty_loss = -(current_penalty * (costs.detach() - cost_budgets)).mean() + budget_loss
                self.logger.info(f'cost_budget: {cost_budgets.mean().item():.4f}, cost_budgets_scaler: {cost_budgets_scaler.mean().item():.4f}, budget_loss: {budget_loss.mean().item():.4f},  mean_task_difficuties: {task_difficuties.mean().item():.4f}, mean_returns: {returns.mean().item():.4f}, mean_costs: {costs.mean().item():.4f}')
                self.logger.info(f'curr_penalty: {current_penalty.mean().item():.4f}, penalty_loss: {penalty_loss.mean().item():.4f},  cost_values: {cost_values.mean().item():.4f}')
            else:
                penalty_loss = torch.zeros(size=(1,), device=self.device).mean()

            loss = actor_loss \
                    + self.coef_critic_loss * critic_loss \
                    + self.coef_cost_critic_loss * cost_critic_loss \
                    - self.coef_entropy_loss * entropy_loss \
                    + self.coef_mask_loss * mask_loss \
                    + prediction_loss \
                    + penalty_loss
            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.rl.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
            self.optimizer.step()

            if self.update_time % self.config.training.log_interval == 0:
                info = {
                    'lr': self.optimizer.defaults['lr'],
                    'loss/loss': loss.detach().cpu().numpy(),
                    'loss/actor_loss': actor_loss.detach().cpu().numpy(),
                    'loss/critic_loss': critic_loss.detach().cpu().numpy(),
                    'loss/cost_critic_loss': cost_critic_loss.detach().cpu().numpy(),
                    'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
                    'loss/penalty_loss': penalty_loss.detach().cpu().numpy(),
                    'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
                    'value/old_action_logprob': old_action_logprobs.mean().cpu().numpy(),
                    'value/value': values.detach().mean().cpu().numpy(),
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'cost_value/cost_value': cost_values.detach().mean().cpu().numpy(),
                    # 'cost_value/cost_advantages': cost_advantages.detach().mean().cpu().numpy(),
                    'cost_value/cost': batch_costs.mean().cpu().numpy(),
                    'penalty_params': cur_penalty.mean().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                self.logger.log(data=info, step=self.update_time)
            self.update_time += 1

        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        if self.config.training.distributed_training: self.sync_parameters()
        return loss.detach()



# def get_gap(solver, batch_index):
#     # get gap ratio
#     surrogate_solver = copy.deepcopy(solver)
#     sm_values = surrogate_solver.estimate_value(observations)
#     sm_current_penalty = surrogate_solver.calc_penalty_params(observations)
#     sm_cost_values = surrogatsurrogate_solvere_model.estimate_cost(observations).detach()
#     # Official
#     # calculate advantage
#     sm_advantages = returns - sm_values.detach()
#     cost_advantages = costs - sm_cost_values.detach()
#     if self.config.rl.norm_advantage and values.numel() != 0:
#         sm_advantages = (sm_advantages - sm_advantages.mean()) / (sm_advantages.std() + 1e-9)
#         sm_cost_advantages = (sm_cost_advantages - sm_cost_advantages.mean()) / (sm_cost_advantages.std() + 1e-9)
#     ratio = torch.exp(action_logprobs - old_action_logprobs)
#     surr1 = ratio * sm_advantages
#     surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
    
#     # calculate loss
#     cur_penalty =  self.calc_penalty_params(observations).detach()
#     actor_loss = - torch.min(surr1, surr2).mean() + (cur_penalty * (ratio * cost_advantages)).mean()
#     critic_loss = self.criterion_critic(returns, values)
#     cost_critic_loss = self.criterion_cost_critic(costs, cost_values)
#     entropy_loss = dist_entropy.mean()

#     # calculate loss
#     actor_loss = - torch.min(surr1, surr2).mean()
#     critic_loss = self.criterion_critic(returns, values)
#     cost_critic_loss = self.criterion_cost_critic(costs, cost_values)
#     entropy_loss = dist_entropy.mean()

#     mask_loss = other.get('mask_actions_probs', 0)
#     prediction_loss = other.get('prediction_loss', 0)


#     loss = sm_actor_loss \
#             + self.coef_critic_loss * csm_ritic_loss \
#             + self.coef_cost_critic_loss * sm_cost_critic_loss \
#             - self.coef_entropy_loss * sm_entropy_loss \
#             + self.coef_mask_loss * sm_mask_loss \
#             + sm_prediction_loss \
#             + sm_penalty_loss

#     sm_objective =  sm_values + sm_current_penalty * sm_cost_values
