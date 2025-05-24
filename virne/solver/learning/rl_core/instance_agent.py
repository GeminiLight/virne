# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import gym
import copy
import time
from sympy import im
import tqdm
import torch
import pprint
import numpy as np

from .buffer import RolloutBuffer


class InstanceAgent(object):
    """Training and Inference Methods for Resource Allocation Agent"""
    def __init__(self, InstanceEnv) -> None:
        self.InstanceEnv = InstanceEnv

    def get_training_seed(self):
        return self.seed if not self.config.training.if_use_random_training_seed else None

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
        solution = self.searcher.find_solution(instance_env)
        return solution

    def validate(self, env, checkpoint_path=None):
        self.logger.info(f"\n{'-' * 20}  Validate  {'-' * 20}\n")
        if checkpoint_path: self.load_model(checkpoint_path)

        pbar = tqdm.tqdm(desc=f'Validate', total=env.v_net_simulator.num_v_nets)
        
        self.eval()
        instance = env.reset(seed=self.seed)
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
        self.logger.info(f"\n{'-' * 20}     Done    {'-' * 20}\n")

    def get_baseline_solution_info(self, instance, if_use_baseline_solver=True):
        """
        Get the baseline solution info for the instance, including:
            - result: whether the baseline solution is feasible
            - v_net_r2c_ratio: the revenue to cost ratio of the baseline solution

        Args:
            instance (dict): the instance to be solved
            if_use_baseline_solver (bool, optional): whether to use the baseline solver. Defaults to True.

        Returns:
            dict: the baseline solution info
        """
        if not if_use_baseline_solver:
            return {
                'result': True,
                'v_net_r2c_ratio': 0,
                'v_net_max_single_step_hard_constraint_violation': 0,
            }
        self.baseline_solver.eval()
        if hasattr(self.baseline_solver, 'unsafe_solve') and self.if_allow_baseline_unsafe_solve:
            baseline_solution = self.baseline_solver.unsafe_solve(instance)
            baseline_solution_info = self.counter.count_solution(instance['v_net'], baseline_solution)
            if baseline_solution.is_feasible():
                baseline_solution_info['result'] = True
            else:
                baseline_solution_info['result'] = False
        else:
            baseline_solution = self.baseline_solver.solve(instance)
            baseline_solution_info = self.counter.count_solution(instance['v_net'], baseline_solution)
            print(f"Baseline - Result {baseline_solution_info['result']}, Violation {baseline_solution_info['v_net_total_hard_constraint_violation']}, Is Feasible {baseline_solution.is_feasible()}")
        # if baseline_solution_info['result'] == 0:
        #     print(baseline_solution_info)
        #     import pdb; pdb.set_trace()
        return baseline_solution_info

    def learn_with_instance_parallelly(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        vectorized_env = gym.vector.SyncVectorEnv([
            lambda: self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
        ]*2)
        instance_obs = vectorized_env.reset()

    def learn_with_instance(self, instance):
        # sub env for sub agent
        import time
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_buffer = RolloutBuffer()
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
        instance_obs = instance_env.reset()
        while True:
            tensor_instance_obs = self.preprocess_obs(instance_obs, self.device)

            time_a = time.time()
            action, action_logprob = self.select_action(tensor_instance_obs, sample=True)
            time_b = time.time()

            if instance_env.curr_v_node_id == 0:
                feature_time = time_b - time_a
                # print(feature_time)

            value = self.estimate_value(tensor_instance_obs)
            next_instance_obs, instance_reward, instance_done, instance_info = instance_env.step(action)
            instance_buffer.add(instance_obs, action, instance_reward, instance_done, action_logprob, value=value, next_obs=next_instance_obs)
            # instance_buffer.action_masks.append(mask if isinstance(mask, np.ndarray) else mask.cpu().numpy())

            if instance_done:
                break

            instance_obs = next_instance_obs

        solution = instance_env.solution
        last_value = self.estimate_value(self.preprocess_obs(next_instance_obs, self.device))
        return solution, instance_buffer, last_value

    def merge_instance_experience(self, instance, solution, instance_buffer, last_value):
        ### -- if_use_negative_sample -- ##
        if self.config.rl.if_use_negative_sample:
            baseline_solution_info = self.get_baseline_solution_info(instance, self.if_use_baseline_solver)
            if baseline_solution_info['result'] or solution['result']:
                instance_buffer.compute_returns_and_advantages(last_value, gamma=self.config.rl.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method)
                self.buffer.merge(instance_buffer)
                self.time_step += 1
            else:
                pass
        elif solution['result']:  #  or True
            instance_buffer.compute_returns_and_advantages(last_value, gamma=self.config.rl.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method)
            # instance_buffer.compute_mc_returns(gamma=self.config.rl.gamma)
            self.buffer.merge(instance_buffer)
            self.time_step += 1
        else:
            pass
        return self.buffer

    def learn_singly(self, env, num_epochs=1, **kwargs):
        # main env
        import time
        start_time = time.time()
        for epoch_id in range(num_epochs):
            self.logger.info(f'Training Epoch: {epoch_id}')
            self.training_epoch_id = epoch_id
            instance = env.reset(seed=self.get_training_seed())
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
                # env.v_net.num_nodes | env.v_net_simulator.v_nets[0].num_nodes
                if self.buffer.size() >= self.target_steps:
                    loss = self.update()

                instance, reward, done, info = env.step(solution)

                if done:
                    break
                
            epoch_logprobs_tensor = np.concatenate(epoch_logprobs, axis=0)
            end_time = time.time()
            time_cost = end_time - start_time
            self.logger.info(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["long_term_r2c_ratio"]:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}, time_cost {time_cost}')
            self.logger.log(
                {
                    'train_epochs/epoch': epoch_id,
                    'train_epochs/success_count': success_count,
                    'train_epochs/r2c': info['long_term_r2c_ratio'],
                    'train_epochs/mean_logprob': epoch_logprobs_tensor.mean(),
                    'train_epochs/time_cost': time_cost
                }, step=epoch_id
            )
            if self.rank == 0:
                # save
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.config.training.save_interval == 0:
                    self.save_model(f'model-{epoch_id}.pkl')
                    self.save_model(f'model.pkl')
                # validate
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.config.training.eval_interval == 0:
                    self.validate(env)