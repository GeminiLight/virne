import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .sub_env import SubEnv
from .net import Actor, ActorCritic, Critic
from ..rl_solver import *
from base import Solution


class PGCNNSolver(PGSolver):

    def __init__(self, 
                 reusable=False,
                 verbose=1,
                 **kwargs):
        super(PGCNNSolver, self).__init__('pg_cnn', reusable, verbose, **kwargs)
        feature_dim = 4  # (n_attrs, e_attrs, dist, degree)
        action_dim = kwargs['pn_setting']['num_nodes']
        self.policy = ActorCritic(feature_dim, action_dim, self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_actor},
        ])
        self.SubEnv = SubEnv
        
    def preprocess_obs(self, observation):
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(self.device)
        return observation

    def preprocess_batch_obs(self, obs_batch):
        r"""Preprocess the observation to adapte to batch mode."""
        observation = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        return observation

    # def learn(self, env, num_epochs=1, start_epoch=0, batch_size=32, save_timestep=1000, config=None):
    #     self.time_step = 0
    #     # main env
    #     for epoch_id in range(start_epoch, start_epoch + num_epochs):
    #         instance = env.reset()
    #         vn, pn = instance['vn'], instance['pn']
    #         success_count = 0
    #         epoch_logprobs = []
    #         for i in range(1000):
    #             ### --- sub env --- ###
    #             sub_env = SubEnv(pn, vn)
    #             sub_obs = sub_env.get_observation()
    #             action_logprob_list = []
    #             for vnf_id in list(vn.nodes):
    #                 mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
    #                 action, action_logprob = self.select_action(sub_obs, mask=mask, sample=True)
    #                 next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action)

    #                 action_logprob_list.append(action_logprob)
    #                 if sub_done:
    #                     break

    #                 sub_obs = next_sub_obs

    #             if sub_env.curr_solution['result']:
    #                 success_count += 1
    #                 self.time_step += 1
    #                 solution = sub_env.curr_solution
    #                 sub_logprob = torch.cat(action_logprob_list, dim=0).mean().unsqueeze(dim=0)
    #                 self.buffer.returns.append(sub_reward)
    #                 self.buffer.logprobs.append(sub_logprob)
    #                 epoch_logprobs.append(sub_logprob)
    #             else:
    #                 solution = Solution(vn)

    #             # update parameters
    #             if self.time_step !=0 and sub_env.curr_solution['result'] and self.time_step % batch_size == 0:
    #                 self.update()
    #             ### --- sub env --- ###

    #             instance, reward, done, info = env.step(solution)

    #             # epoch finished
    #             if not done:
    #                 vn, pn = instance['vn'], instance['pn']
    #             else:
    #                 break
    #         epoch_logprobs_tensor = torch.cat(epoch_logprobs, dim=0)
    #         print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["total_r2c"]:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}')
    #         self.save_model(f'model-{epoch_id}.pkl')