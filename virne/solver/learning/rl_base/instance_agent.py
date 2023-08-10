# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import gym
import copy
import time
import tqdm
import torch
import pprint
import numpy as np

from .buffer import RolloutBuffer


class InstanceAgent(object):
    """Training and Inference Methods for Resource Allocation Agent"""
    def __init__(self, InstanceEnv) -> None:
        self.InstanceEnv = InstanceEnv

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        solution = self.searcher.find_solution(instance_env)
        return solution

    def validate(self, env, checkpoint_path=None):
        print(f"\n{'-' * 20}  Validate  {'-' * 20}\n") if self.verbose >= 0 else None
        if checkpoint_path is not None: self.load_model(checkpoint_path)

        pbar = tqdm.tqdm(desc=f'Validate', total=env.num_v_nets) if self.verbose <= 1 else None
        
        self.eval()
        instance = env.reset()
        while True:
            solution = self.solve(instance)
            next_instance, _, done, info = env.step(solution)

            if pbar is not None: 
                pbar.update(1)
                pbar.set_postfix({
                    'ac': f'{info["success_count"] / info["v_net_count"]:1.2f}',
                    'r2c': f'{info["long_term_r2c_ratio"]:1.2f}',
                    'inservice': f'{info["inservice_count"]:05d}',
                })
            if done:
                break
            instance = next_instance

        if pbar is not None: pbar.close()
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

    def learn_with_instance_parallelly(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        vectorized_env = gym.vector.SyncVectorEnv([
            lambda: self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        ]*2)
        instance_obs = vectorized_env.reset()

    def learn_with_instance(self, instance):
        # sub env for sub agent
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_buffer = RolloutBuffer()
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        instance_obs = instance_env.reset()
        while True:
            tensor_instance_obs = self.preprocess_obs(instance_obs, self.device)
            action, action_logprob = self.select_action(tensor_instance_obs, sample=True)
            value = self.estimate_value(tensor_instance_obs)
            next_instance_obs, instance_reward, instance_done, instance_info = instance_env.step(action)
            instance_buffer.add(instance_obs, action, instance_reward, instance_done, action_logprob, value=value)
            # instance_buffer.action_masks.append(mask if isinstance(mask, np.ndarray) else mask.cpu().numpy())

            if instance_done:
                break

            instance_obs = next_instance_obs

        solution = instance_env.solution
        last_value = self.estimate_value(self.preprocess_obs(next_instance_obs, self.device))
        return solution, instance_buffer, last_value

    def merge_instance_experience(self, instance, solution, instance_buffer, last_value):
        ### -- use_negative_sample -- ##
        if self.use_negative_sample:
            baseline_solution_info = self.get_baseline_solution_info(instance, self.use_baseline_solver)
            if baseline_solution_info['result'] or solution['result']:
                instance_buffer.compute_returns_and_advantages(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method)
                self.buffer.merge(instance_buffer)
                self.time_step += 1
            else:
                pass
        elif solution['result']:  #  or True
            instance_buffer.compute_mc_returns(gamma=self.gamma)
            self.buffer.merge(instance_buffer)
            self.time_step += 1
        else:
            pass
        return self.buffer

    def learn_singly(self, env, num_epochs=1, **kwargs):
        # main env
        for epoch_id in range(num_epochs):
            print(f'Training Epoch: {epoch_id}') if self.verbose > 0 else None
            self.training_epoch_id = epoch_id

            instance = env.reset()
            success_count = 0
            epoch_logprobs = []
            revenue2cost_list = []
            while True:
                ### --- instance-level --- ###
                solution, instance_buffer, last_value = self.learn_with_instance(instance)
                epoch_logprobs += instance_buffer.logprobs
                self.merge_instance_experience(instance, solution, instance_buffer, last_value)

                if solution.is_feasible():
                    success_count += 1
                    revenue2cost_list.append(solution['v_net_r2c_ratio'])
                # update parameters
                if self.buffer.size() >= self.target_steps:
                    loss = self.update()

                instance, reward, done, info = env.step(solution)

                if done:
                    break
                
            epoch_logprobs_tensor = np.concatenate(epoch_logprobs, axis=0)
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["long_term_r2c_ratio"]:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}') if self.verbose > 0 else None
            if self.rank == 0:
                # save
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.save_interval == 0:
                    self.save_model(f'model-{epoch_id}.pkl')
                # validate
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.eval_interval == 0:
                    self.validate(env)