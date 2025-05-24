# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import time
import tqdm
import torch
import pprint
import numpy as np

from .instance_agent import InstanceAgent
from .buffer import RolloutBufferWithCost


class SafeInstanceAgent(InstanceAgent):

    def __init__(self, InstanceEnv):
        super(SafeInstanceAgent, self).__init__(InstanceEnv)
        self.compute_advantage_method = 'mc'

    def unsafe_solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
        instance_env.if_allow_constraint_violation = True
        solution = self.searcher.find_solution(instance_env)
        return solution

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
        instance_env.if_allow_constraint_violation = False
        solution = self.searcher.find_solution(instance_env)
        return solution

    def learn_singly(self, env, num_epochs=1, **kwargs):
        # main env
        import time
        start_time = time.time()
        for epoch_id in range(num_epochs):
            self.logger.info(f'Training Epoch: {epoch_id}')
            seed = self.seed if not self.config.training.if_use_random_training_seed else None
            instance = env.reset(seed=self.get_training_seed())
            success_count = 0
            epoch_logprobs = []
            epoch_cost_list = []
            revenue2cost_list = []
            cost_list = []
            import time
            for i in range(env.v_net_simulator.num_v_nets):
                ### --- sub env --- ###
                sub_buffer = RolloutBufferWithCost()
                v_net, p_net = instance['v_net'], instance['p_net']
                instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
                instance_obs = instance_env.get_observation()
                instance_done = False
                while not instance_done:
                    tensor_instance_obs = self.preprocess_obs(instance_obs, self.device)

                    time_a = time.time()
                    action, action_logprob = self.select_action(tensor_instance_obs, sample=True)
                    time_b = time.time()

                    if instance_env.curr_v_node_id == 0:
                        feature_time = time_b - time_a
                        print(feature_time)

                    value = self.estimate_value(tensor_instance_obs) if hasattr(self.policy, 'evaluate') else None
                    cost_value = self.estimate_cost(tensor_instance_obs) if hasattr(self.policy, 'evaluate_cost') else None
                    next_instance_obs, instance_reward, instance_done, instance_info = instance_env.step(action)
                    sub_buffer.add(instance_obs, action, instance_reward, instance_done, action_logprob, value=value)
                    sub_buffer.costs.append(instance_env.solution['v_net_single_step_hard_constraint_offset'])
                    sub_buffer.cost_values.append(cost_value)
                    cost_list.append(instance_env.solution['v_net_single_step_hard_constraint_offset'])
                    if instance_done:
                        break
                    instance_obs = next_instance_obs
                last_value = self.estimate_value(self.preprocess_obs(next_instance_obs, self.device)) if hasattr(self.policy, 'evaluate') else None
                solution = instance_env.solution
                # print(f'{v_net.num_nodes:2d}', f'{sum(sub_buffer.costs):2.2f}', f'{sum(sub_buffer.costs)/ v_net.num_nodes:2.2f}', sub_buffer.costs)
                epoch_logprobs += sub_buffer.logprobs
                self.merge_instance_experience(instance, solution, sub_buffer, last_value)
                # instance_env.solution['result'] or self.config.rl.if_use_negative_sample:  #  or True
                if solution.is_feasible():
                    success_count = success_count + 1
                    revenue2cost_list.append(solution['v_net_r2c_ratio'])
                # update parameters
                if self.buffer.size() >= self.target_steps:
                    avg_cost = sum(cost_list) / len(cost_list)
                    loss = self.update(avg_cost)
                    epoch_cost_list += cost_list
                    self.logger.info(f'avg_cost: {avg_cost:+2.4f}, cost budget: {self.cost_budget:+2.4f}, loss: {loss.item():+2.4f}, mean r2c: {np.mean(revenue2cost_list):+2.4f}')
                ### --- sub env --- ###
                instance, reward, done, info = env.step(solution)
                # epoch finished
                if done:
                    break
            epoch_logprobs_tensor = np.concatenate(epoch_logprobs, axis=0)
            avg_epoch_cost = sum(epoch_cost_list) / len(epoch_cost_list)
            end_time = time.time()
            time_cost = end_time - start_time
            self.logger.info(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["long_term_r2c_ratio"]:1.4f}, avg_cost: {avg_epoch_cost:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}, time_cost {time_cost}')
            self.logger.log(
                {
                    'train_epochs/epoch': epoch_id,
                    'train_epochs/success_count': success_count,
                    'train_epochs/r2c': info['long_term_r2c_ratio'],
                    'train_epochs/avg_cost': avg_epoch_cost,
                    'train_epochs/mean_logprob': epoch_logprobs_tensor.mean(),
                    'train_epochs/time_cost': time_cost
                }, step=epoch_id
            )
            if self.rank == 0:
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.config.training.save_interval == 0:
                    self.save_model(f'model-{epoch_id}.pkl')
                    self.save_model(f'model.pkl')
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.config.training.eval_interval == 0:
                    self.validate(env)

    def merge_instance_experience(self, instance, solution, instance_buffer, last_value):
        merge_flag = False
        if self.config.rl.if_use_negative_sample:
            baseline_solution_info = self.get_baseline_solution_info(instance, self.if_use_baseline_solver)
            if baseline_solution_info['result'] or solution['result']:
                merge_flag = True
        elif solution['result']:
            merge_flag = True
        else:
            pass
        if merge_flag:
            instance_buffer.compute_returns_and_advantages(last_value, gamma=self.config.rl.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method)
            instance_buffer.compute_cost_returns(gamma=self.config.rl.gamma, method=self.compute_cost_method)
            self.buffer.merge(instance_buffer)
            self.time_step += 1
        return self.buffer