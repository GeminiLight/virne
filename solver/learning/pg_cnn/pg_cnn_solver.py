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


class PgCnnSolver(PPOSolver):

    name = 'pg_cnn'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(PgCnnSolver, self).__init__(controller, recorder, counter, **kwargs)
        feature_dim = 4  # (n_attrs, e_attrs, dist, degree)
        action_dim = kwargs['p_net_setting']['num_nodes']
        self.policy = ActorCritic(feature_dim, action_dim, self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_actor},
        ])
        self.SubEnv = SubEnv
        self.preprocess_obs = obs_as_tensor
        
def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, list):
        obs_batch = obs
        r"""Preprocess the observation to adapte to batch mode."""
        observation = torch.FloatTensor(np.array(obs_batch)).to(device)
        return observation
    # batch
    else:
        observation = obs
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(device)
        return observation


    # def learn(self, env, num_epochs=1, start_epoch=0, batch_size=32, save_timestep=1000, config=None):
    #     self.time_step = 0
    #     # main env
    #     for epoch_id in range(start_epoch, start_epoch + num_epochs):
    #         instance = env.reset()
    #         v_net, p_net = instance['v_net'], instance['p_net']
    #         success_count = 0
    #         epoch_logprobs = []
    #         for i in range(1000):
    #             ### --- sub env --- ###
    #             sub_env = SubEnv(p_net, v_net)
    #             sub_obs = sub_env.get_observation()
    #             action_logprob_list = []
    #             for v_node_id in list(v_net.nodes):
    #                 mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
    #                 action, action_logprob = self.select_action(sub_obs, mask=mask, sample=True)
    #                 next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action)

    #                 action_logprob_list.append(action_logprob)
    #                 if sub_done:
    #                     break

    #                 sub_obs = next_sub_obs

    #             if sub_env.solution['result']:
    #                 success_count += 1
    #                 self.time_step += 1
    #                 solution = sub_env.solution
    #                 sub_logprob = torch.cat(action_logprob_list, dim=0).mean().unsqueeze(dim=0)
    #                 self.buffer.returns.append(sub_reward)
    #                 self.buffer.logprobs.append(sub_logprob)
    #                 epoch_logprobs.append(sub_logprob)
    #             else:
    #                 solution = Solution(v_net)

    #             # update parameters
    #             if self.time_step !=0 and sub_env.solution['result'] and self.time_step % batch_size == 0:
    #                 self.update()
    #             ### --- sub env --- ###

    #             instance, reward, done, info = env.step(solution)

    #             # epoch finished
    #             if not done:
    #                 v_net, p_net = instance['v_net'], instance['p_net']
    #             else:
    #                 break
    #         epoch_logprobs_tensor = torch.cat(epoch_logprobs, dim=0)
    #         print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["total_r2c"]:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}')
    #         self.save_model(f'model-{epoch_id}.pkl')