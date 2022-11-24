import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


from base import Solution
from .sub_env import SubEnv
from .net import Actor
from ..rl_solver import RLSolver, PGSolver


class PGSeq2SeqSolver(PGSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(PGSeq2SeqSolver, self).__init__('pg_cnn', controller, recorder, counter, **kwargs)
        feature_dim = 3  # (n_attrs, e_attrs, dist, degree)
        action_dim = 100
        self.policy = Actor(feature_dim, action_dim, self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_actor},
        ])

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = SubEnv(p_net, v_net)
        SOS = torch.IntTensor(np.array([p_net.num_nodes])).to(self.device)
        sub_obs = sub_env.get_observation()
        input = SOS
        outputs, hidden = self.policy.encode(self.preprocess_obs(sub_obs))
        for v_node_id in list(v_net.nodes):
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            action, action_logprob, hidden = self.select_action(input, hidden, mask=mask, sample=False)
            _, reward, done, info = sub_env.step(action)
            input = torch.IntTensor(np.array([action])).to(self.device)
            if done:
                return sub_env.solution

    def preprocess_obs(self, observation):
        observation = torch.FloatTensor(observation).unsqueeze(dim=0).to(self.device)
        return observation

    def preprocess_batch_obs(self, obs_batch):
        r"""Preprocess the observation to adapte to batch mode."""
        observation = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        return observation

    def select_action(self, input, hidden, mask=None, sample=True):
        action_logits, hidden = self.policy.decode(input, hidden)

        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        if mask is not None:
            mask = torch.IntTensor(mask).to(self.device).expand_as(action_logits)
            candicate_action_logits = action_logits.detach() + mask.log()
        else:
            candicate_action_logits = action_logits.detach()

        if sample:
            candicate_dist = Categorical(logits=candicate_action_logits)
            action = candicate_dist.sample()
        else:
            action = candicate_action_logits.argmax(-1)
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, hidden

    def learn(self, env, num_epochs=1, start_epoch=0, batch_size=32, save_timestep=1000, config=None):
        self.time_step = 0
        # main env
        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            instance = env.reset()
            v_net, p_net = instance['v_net'], instance['p_net']
            SOS = torch.IntTensor(np.array([p_net.num_nodes])).to(self.device)
            success_count = 0
            epoch_logprobs = []
            for i in range(2000):
                ### --- sub env --- ###
                solution = Solution(v_net)
                sub_env = SubEnv(p_net, v_net)
                sub_obs = sub_env.get_observation()
                action_logprob_list = []

                outputs, hidden = self.policy.encode(self.preprocess_obs(sub_obs))
                input = SOS
                for v_node_id in list(v_net.nodes):
                    mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
                    action, action_logprob, hidden = self.select_action(input, hidden, mask=mask, sample=True)
                    _, sub_reward, sub_done, sub_info = sub_env.step(action)

                    input = torch.IntTensor(np.array([action])).to(self.device)
                    action_logprob_list.append(action_logprob)
                    if sub_done:
                        break

                if sub_env.solution['result']:
                    success_count += 1
                    self.time_step += 1
                    solution = sub_env.solution
                    sub_logprob = torch.cat(action_logprob_list, dim=0).mean().unsqueeze(dim=0)
                    self.buffer.rewards.append(sub_reward)
                    self.buffer.logprobs.append(sub_logprob)
                    epoch_logprobs.append(sub_logprob)

                else:
                    solution = Solution(v_net)

                # update parameters
                if self.time_step !=0 and sub_env.solution['result'] and self.time_step % batch_size == 0:
                    self.update()
                ### --- sub env --- ###

                instance, reward, done, info = env.step(solution)

                # epoch finished
                if not done:
                    v_net, p_net = instance['v_net'], instance['p_net']
                else:
                    break
            epoch_logprobs_tensor = torch.cat(epoch_logprobs, dim=0)
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["total_r2c"]:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}')
            self.save_model(f'model-{epoch_id}.pkl')