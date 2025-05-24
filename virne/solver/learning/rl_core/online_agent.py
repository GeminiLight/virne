# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import time
import tqdm
import torch
import numpy as np


class OnlineAgent(object):
    """Training and Inference Methods for Admission Control Agent"""
    def __init__(self) -> None:
        pass

    def solve(self, instance):
        instance = self.preprocess_obs(instance, self.device)
        action, action_logprob = self.select_action(instance, sample=False)
        return action

    def learn_singly(self, env, num_epochs=1, **kwargs):
        # main env
        for epoch_id in range(num_epochs):
            obs = env.reset()
            success_count = 0
            for i in range(env.v_net_simulator.num_v_nets):
                tensor_obs = self.preprocess_obs(obs, self.device)
                action, action_logprob = self.select_action(tensor_obs, sample=True)
                value = self.estimate_value(tensor_obs)
                next_obs, reward, done, info = env.step(action)
                self.buffer.add(obs, action, reward, done, action_logprob, value=value)
                obs = next_obs
                self.time_step += 1
                # update parameters
                if done and self.buffer.size() >= self.batch_size:  # done and 
                    with torch.no_grad():
                        last_value = float(self.estimate_value(self.preprocess_obs(next_obs, self.device)).detach()[0]) if not done else 0.
                    # if self.config.rl.norm_reward:
                        # self.running_stats.update(self.buffer.rewards)
                        # self.buffer.rewards = ((np.array(self.buffer.rewards) - self.running_stats.mean) / (np.sqrt(self.running_stats.var + 1e-9))).tolist()
                    self.buffer.compute_returns_and_advantages(last_value, gamma=self.config.rl.gamma, gae_lambda=self.gae_lambda, method=self.compute_return_method)
                    loss = self.update()
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["long_term_r2c_ratio"]:1.4f}, {self.running_stats.mean}-{np.sqrt(self.running_stats.var)}')
            if self.rank == 0:
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.config.training.eval_interval == 0:
                    self.validate(env)
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.config.training.save_interval == 0:
                    self.save_model(f'model-{epoch_id}.pkl')

    def validate(self, env, checkpoint_path=None):
        self.logger.info(f"\n{'-' * 20}  Validate  {'-' * 20}\n")
        if checkpoint_path is not None: self.load_model(checkpoint_path)

        pbar = tqdm.tqdm(desc=f'Validate', total=env.v_net_simulator.num_v_nets)
        
        instance = env.reset(0)
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
