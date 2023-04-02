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
from .buffer import RolloutBuffer
from .utils import apply_mask_to_logit, get_observations_sample


class SafeInstanceAgent(InstanceAgent):

    def __init__(self):
        super(SafeInstanceAgent, self).__init__()

    def learn(self, env, num_epochs=1, start_epoch=0, save_timestep=1000, config=None, **kwargs):
        # main env
        self.start_time = time.time()
        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            print(f'Training Epoch: {epoch_id}')
            instance = env.reset()
            success_count = 0
            epoch_logprobs = []
            epoch_cost_list = []
            revenue2cost_list = []
            cost_list = []
            for i in range(env.num_v_nets):
                ### -- baseline -- ##
                baseline_solution_info = self.get_baseline_solution_info(instance, self.use_baseline_solver)
                ### --- sub env --- ###
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
                    norm_current_violation = sub_info['current_violation'] * 0.1
                    sub_buffer.costs.append(norm_current_violation)
                    cost_list.append(norm_current_violation)
                    if sub_done:
                        break

                    sub_obs = next_sub_obs
                    
                solution = sub_env.solution
                # sub_env.solution['result'] or self.use_negative_sample:  #  or True
                if sub_env.solution.is_feasible(): success_count = success_count + 1 
                if self.use_negative_sample:
                    if baseline_solution_info['result'] or sub_env.solution['result']:
                        revenue2cost_list.append(sub_reward)
                        # sub_logprob = torch.cat(sub_logprob_list, dim=0).mean().unsqueeze(dim=0)
                        sub_buffer.compute_mc_returns(gamma=self.gamma)
                        self.buffer.merge(sub_buffer)
                        epoch_logprobs += sub_buffer.logprobs
                        self.time_step += 1
                    else:
                        pass
                elif sub_env.solution['result']:  #  or True
                    revenue2cost_list.append(sub_reward)
                    # sub_logprob = torch.cat(sub_logprob_list, dim=0).mean().unsqueeze(dim=0)
                    sub_buffer.compute_mc_returns(gamma=self.gamma)
                    self.buffer.merge(sub_buffer)
                    epoch_logprobs += sub_buffer.logprobs
                    self.time_step += 1
                else:
                    pass

                # update parameters
                if self.buffer.size() >= self.target_steps:
                    avg_cost = sum(cost_list) / len(cost_list)
                    loss = self.update(avg_cost)
                    epoch_cost_list += cost_list
                    print(f'avg_cost: {avg_cost:+2.4f}, loss: {loss.item():+2.4f}, mean r2c: {np.mean(revenue2cost_list):+2.4f}')

                ### --- sub env --- ###
                instance, reward, done, info = env.step(solution)
                # instance = env.reset()
                # epoch finished
                if done:
                    break
            summary_info = env.summary_records()
            epoch_logprobs_tensor = torch.cat(epoch_logprobs, dim=0)
            avg_epoch_cost = sum(epoch_cost_list) / len(epoch_cost_list)
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["total_r2c"]:1.4f}, avg_cost: {avg_epoch_cost:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}')
            self.save_model(f'model-{epoch_id}.pkl')
            # validate
            if (epoch_id + 1) != (start_epoch + num_epochs) and (epoch_id + 1) % self.eval_interval == 0:
                self.validate(env)

        self.end_time = time.time()
        print(f'\nTotal training time: {(self.end_time - self.start_time) / 3600:4.6f} h')


class SafeRLSolver(RLSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(SafeRLSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.penalty_params = torch.tensor(1.0, requires_grad=True, device=self.device).float()
        self.lr_cost_critic = kwargs.get('lr_cost_critic', self.lr_critic)
        self.lr_penalty_params = kwargs.get('lr_penalty_params', self.lr_critic * 10)
        self.coef_cost_critic_loss = kwargs.get('lr_penalty_params', self.coef_critic_loss)
        self.criterion_cost_critic = nn.MSELoss()
        self.cost_budget = 0.02

    def save_model(self, checkpoint_fname):
        checkpoint_fname = os.path.join(self.model_dir, checkpoint_fname)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'penalty_params': self.penalty_params
        }, checkpoint_fname)
        print(f'Save model to {checkpoint_fname}\n') if self.verbose >= 0 else None

    def load_model(self, checkpoint_fname):
        super().load_model(checkpoint_fname)
        checkpoint = torch.load(checkpoint_fname)
        if 'penalty_params' in checkpoint:
            self.penalty_params = checkpoint['penalty_params']

    def evaluate_costs(self, observations):
        cost_values = self.policy.cost_critic(observations).squeeze(-1)
        return cost_values


class FixedPenaltyPPOSolver(PPOSolver, SafeRLSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(FixedPenaltyPPOSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.penalty_params.requires_grad = False

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size

        batch_observations = self.preprocess_obs(self.buffer.observations, self.device)
        batch_actions = torch.LongTensor(self.buffer.actions).to(self.device)
        batch_old_action_logprobs = torch.cat(self.buffer.logprobs, dim=0).to(self.device).detach()
        batch_rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        batch_costs = torch.FloatTensor(self.buffer.costs).to(self.device)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns).to(self.device)

        batch_returns = torch.FloatTensor(self.buffer.returns).to(self.device)

        if len(self.buffer.action_masks) != 0 and self.mask_actions:
            batch_masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device)
        else:
            batch_masks = None

        # update the policy params repeatly
        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long().to(self.device)
            observations  = get_observations_sample(batch_observations, sample_indices)
            actions, returns, cost_returns = batch_actions[sample_indices], batch_returns[sample_indices], batch_cost_returns[sample_indices]
            old_action_logprobs = batch_old_action_logprobs[sample_indices]
            masks = batch_masks[sample_indices] if batch_masks is not None else None
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
            
            # calculate advantage
            advantages = returns - cost_returns * self.penalty_params - values.detach()
            if self.norm_advantage and values.numel() != 0:
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
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'cost_value/cost_return': cost_returns.mean().cpu().numpy(),
                    'cost_value/cost': batch_costs.mean().cpu().numpy(),
                    'penalty_params': self.penalty_params.detach().cpu().numpy(),
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


class LagrangianPPOSolver(PPOSolver, SafeRLSolver):

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        batch_observations = self.preprocess_obs(self.buffer.observations, self.device)
        batch_actions = torch.LongTensor(self.buffer.actions).to(self.device)
        batch_old_action_logprobs = torch.cat(self.buffer.logprobs, dim=0).to(self.device).detach()
        batch_rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        batch_costs = torch.FloatTensor(self.buffer.costs).to(self.device)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns).to(self.device)

        batch_returns = torch.FloatTensor(self.buffer.returns).to(self.device)

        if len(self.buffer.action_masks) != 0 and self.mask_actions:
            batch_masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device)
        else:
            batch_masks = None

        # only optimize the penalty param once
        # import pdb; pdb.set_trace();
        penalty_loss = - self.penalty_params * (avg_cost - self.cost_budget)
        self.optimizer.zero_grad()
        # print('gradient: ', penalty_loss.grad)
        penalty_loss.backward(retain_graph=True)
        self.optimizer.step()
        cur_penalty = softplus(self.penalty_params).item()
        print('loss: ', penalty_loss.item(), 'penalty_params: ', self.penalty_params, 'avg_cost: ', avg_cost)
        # update the policy params repeatly
        sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long().to(self.device)
            observations  = get_observations_sample(batch_observations, sample_indices)
            actions, returns, cost_returns = batch_actions[sample_indices], batch_returns[sample_indices], batch_cost_returns[sample_indices]
            old_action_logprobs = batch_old_action_logprobs[sample_indices]
            masks = batch_masks[sample_indices] if batch_masks is not None else None
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
            cost_values = self.evaluate_costs(observations)
            
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = cost_returns - cost_values.detach()
            if self.norm_advantage and values.numel() != 0:
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
            if self.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
            if self.open_tb and self.update_time % self.log_interval == 0:
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
                only_tb = not (i == sample_times-1)
                self.log(info, self.update_time, only_tb=only_tb)

            self.update_time += 1

        # print(f'loss: {loss.detach():+2.4f} = {actor_loss.detach():+2.4f} & {critic_loss:+2.4f} & {entropy_loss:+2.4f} & {mask_loss:+2.4f}, ' +
        #         f'action log_prob: {action_logprobs.mean():+2.4f} (old: {batch_old_action_logprobs.detach().mean():+2.4f}), ' +
        #         f'mean reward: {returns.detach().mean():2.4f}', file=self.fwriter) if self.verbose >= 0 else None
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        return loss.detach()


class FeasiblePPOSolver(PPOSolver, SafeRLSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(FeasiblePPOSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.srl_alpha = kwargs.get('srl_alpha', 1.)
        self.srl_beta = kwargs.get('srl_beta', 0.1)

    def calc_penalty_params(self, observations):
        cost_values = self.policy.lambda_net(observations).squeeze(-1)
        return F.softplus(cost_values)

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        batch_observations = self.preprocess_obs(self.buffer.observations, self.device)
        batch_actions = torch.LongTensor(self.buffer.actions).to(self.device)
        batch_old_action_logprobs = torch.cat(self.buffer.logprobs, dim=0).to(self.device).detach()
        batch_rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        batch_costs = torch.FloatTensor(self.buffer.costs).to(self.device)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns).to(self.device)

        batch_returns = torch.FloatTensor(self.buffer.returns).to(self.device)

        if len(self.buffer.action_masks) != 0 and self.mask_actions:
            batch_masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device)
        else:
            batch_masks = None

        # update the policy params repeatly
        # sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        sample_times = self.repeat_times
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long().to(self.device)
            observations  = get_observations_sample(batch_observations, sample_indices)
            actions, returns, cost_returns = batch_actions[sample_indices], batch_returns[sample_indices], batch_cost_returns[sample_indices]
            old_action_logprobs = batch_old_action_logprobs[sample_indices]
            masks = batch_masks[sample_indices] if batch_masks is not None else None
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
            cost_values = self.evaluate_costs(observations)
            
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = cost_returns - cost_values.detach()
            if self.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            
            # calculate loss
            cur_penalty =  self.calc_penalty_params(observations).detach()
            actor_loss = - torch.min(surr1, surr2).mean() + (cur_penalty * (ratio * cost_advantages)).mean()
            critic_loss = self.criterion_critic(returns, values)
            cost_critic_loss = self.criterion_cost_critic(cost_returns, cost_values)
            entropy_loss = dist_entropy.mean()

            mask_loss = other.get('mask_actions_probs', 0)
            prediction_loss = other.get('prediction_loss', 0)

            if (self.update_time+1) % (self.repeat_times) == 0:
                # Compute lam_net loss
                current_penalty = self.calc_penalty_params(observations)
                cost_values = self.evaluate_costs(observations).detach()
                
                # zero_cost_budget = torch.zeros_like(task_difficuties, device=self.device)
                # adaptive_cost_budget = cost_values * (1 - torch.clip(task_difficuties, max=0.1))

                alpha = self.srl_alpha
                beta = self.srl_beta
                min_safety_budget = 0.3
                print('task_difficuty_threshold: ', alpha, 'safe_budget_scaler: ', beta)
                task_difficuties = torch.tanh(cost_values.detach() / values.detach())
                final_safe_budget_scaler = (1 + beta * (task_difficuties - alpha))
                # final_safe_budget_scaler = torch.clip(final_safe_budget_scaler, max=1.)
                # print('final_safe_budget_scaler', final_safe_budget_scaler.mean().item())
                # print('cost_values', cost_values.mean().item())
                original_cost_budget = cost_values.clone()
                scaled_cost_budget = cost_values * final_safe_budget_scaler
                cost_budget = torch.where(cost_values > min_safety_budget, scaled_cost_budget, original_cost_budget)
                cost_budget = cost_values
                # print('scaled_cost_budget', scaled_cost_budget.mean().item())
                # print('cost_budget', cost_budget.mean().item())
                # print('cost_budget>min_safety_budget', (cost_budget>min_safety_budget).float().mean())
                # cost_budget = torch.where(task_difficuties <= 0.6, adaptive_cost_budget, cost_values * (1))
                # cost_budget = torch.where(task_difficuties <= 0.6, adaptive_cost_budget, cost_values * (1 + 0.1))
                # cost_budget = cost_values.detach() * 0.95
                # cost_budget = self.cost_budget = 0.8
                # penalty_loss = -(current_penalty * (cost_values - cost_budget)).mean()
                penalty_loss = -(current_penalty * (cost_returns - cost_budget)).mean()
                # penalty_loss = -(current_penalty * (cost_returns - cost_values)).mean()
                # penalty_loss = -(current_penalty * (cost_values - self.cost_budget)).mean()
                # final_safe_budget_scaler = (task_difficuties - alpha).max()
                print(f'cost_budget: {cost_budget.mean().item():.4f}, budget_scaler: {final_safe_budget_scaler.mean().item():.4f},  mean_task_difficuties: {task_difficuties.mean().item():.4f}, mean_returns: {returns.mean().item():.4f}, mean_cost_returns: {cost_returns.mean().item():.4f}')
                print(f'curr_penalty: {current_penalty.mean().item():.4f}, penalty_loss: {penalty_loss.mean().item():.4f},  cost_values: {cost_values.mean().item():.4f}')
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
            if self.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
            if self.open_tb and self.update_time % self.log_interval == 0:
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
                    'penalty_params': cur_penalty.mean().cpu().numpy(),
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



class AdaptiveStateWiseSafePPOSolver(PPOSolver, SafeRLSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(AdaptiveStateWiseSafePPOSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.srl_alpha = kwargs.get('srl_alpha', 1.)
        self.srl_beta = kwargs.get('srl_beta', 0.1)

    def calc_penalty_params(self, observations):
        cost_values = self.policy.lambda_net(observations).squeeze(-1)
        return torch.clip(F.softplus(cost_values, beta=1, threshold=20), max=10)

    def estimate_budgets(self, observations):
        cost_budgets = self.policy.budget_net(observations).squeeze(-1)
        return F.softplus(cost_budgets, beta=1, threshold=20)

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        batch_observations = self.preprocess_obs(self.buffer.observations, self.device)
        batch_actions = torch.LongTensor(self.buffer.actions).to(self.device)
        batch_old_action_logprobs = torch.cat(self.buffer.logprobs, dim=0).to(self.device).detach()
        batch_rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        batch_costs = torch.FloatTensor(self.buffer.costs).to(self.device)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns).to(self.device)
        batch_costs = batch_cost_returns
        batch_returns = torch.FloatTensor(self.buffer.returns).to(self.device)

        if len(self.buffer.action_masks) != 0 and self.mask_actions:
            batch_masks = torch.IntTensor(np.concatenate(self.buffer.action_masks, axis=0)).to(self.device)
        else:
            batch_masks = None

        # update the policy params repeatly
        # sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        sample_times = self.repeat_times
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long().to(self.device)
            observations  = get_observations_sample(batch_observations, sample_indices)
            actions, returns, costs = batch_actions[sample_indices], batch_returns[sample_indices], batch_costs[sample_indices]
            old_action_logprobs = batch_old_action_logprobs[sample_indices]
            masks = batch_masks[sample_indices] if batch_masks is not None else None
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, masks=masks, return_others=True)
            
            
            cost_values = self.evaluate_costs(observations)
            
            # calculate advantage
            # cur_penalty =  self.calc_penalty_params(observations).detach()
            # safety_budgets =  self.estimate_budgets(observations).detach()
            # return_with_costs = returns - cur_penalty * (costs - safety_budgets)
            # return_with_cost_values = values - cur_penalty * cur_penalty * (cost_values - safety_budgets)
            # advantages = return_with_costs - return_with_cost_values.detach()
            # if self.norm_advantage and values.numel() != 0:
            #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            # ratio = torch.exp(action_logprobs - old_action_logprobs)
            # surr1 = ratio * advantages
            # surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            
            # Official
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = costs - cost_values.detach()
            if self.norm_advantage and values.numel() != 0:
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
                values = self.estimate_obs(observations)
                current_penalty = self.calc_penalty_params(observations)
                cost_values = self.evaluate_costs(observations).detach()
                # current objective
                curr_objective =  returns - current_penalty * costs
                print(curr_objective.mean())

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
                print(f'cost_budget: {cost_budgets.mean().item():.4f}, cost_budgets_scaler: {cost_budgets_scaler.mean().item():.4f}, budget_loss: {budget_loss.mean().item():.4f},  mean_task_difficuties: {task_difficuties.mean().item():.4f}, mean_returns: {returns.mean().item():.4f}, mean_costs: {costs.mean().item():.4f}')
                print(f'curr_penalty: {current_penalty.mean().item():.4f}, penalty_loss: {penalty_loss.mean().item():.4f},  cost_values: {cost_values.mean().item():.4f}')
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
            if self.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.open_tb and self.update_time % self.log_interval == 0:
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
                only_tb = not (i == sample_times-1)
                self.log(info, self.update_time, only_tb=only_tb)
            self.update_time += 1
        # print(f'loss: {loss.detach():+2.4f} = {actor_loss.detach():+2.4f} & {critic_loss:+2.4f} & {entropy_loss:+2.4f} & {mask_loss:+2.4f}, ' +
        #         f'action log_prob: {action_logprobs.mean():+2.4f} (old: {batch_old_action_logprobs.detach().mean():+2.4f}), ' +
        #         f'mean reward: {returns.detach().mean():2.4f}', file=self.fwriter) if self.verbose >= 0 else None
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        return loss.detach()



# def get_gap(solver, batch_index):
#     # get gap ratio
#     surrogate_solver = copy.deepcopy(solver)
#     sm_values = surrogate_solver.estimate_obs(observations)
#     sm_current_penalty = surrogate_solver.calc_penalty_params(observations)
#     sm_cost_values = surrogatsurrogate_solvere_model.evaluate_costs(observations).detach()
#     # Official
#     # calculate advantage
#     sm_advantages = returns - sm_values.detach()
#     cost_advantages = costs - sm_cost_values.detach()
#     if self.norm_advantage and values.numel() != 0:
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
