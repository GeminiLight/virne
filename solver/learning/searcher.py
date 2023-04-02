import os
import copy
import time
import torch
import random
import multiprocessing
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .utils import apply_mask_to_logit


def get_searcher(decode_strategy, policy, preprocess_obs_func, k, device, mask_actions, maskable_policy):
    if decode_strategy in [0, 'random']:
        SearcherClass = RandomSearcher
    elif decode_strategy in [1, 'greedy']:
        SearcherClass = GreedySearcher
    elif decode_strategy in [1, 'sample', 'sampling']:
        SearcherClass = SampleSearcher
    elif decode_strategy in [2, 'beam', 'beam_search']:
        SearcherClass = BeamSearcher
    elif decode_strategy in [3, 'recovable']:
        SearcherClass = RecovableSearcher
    else:
        raise NotImplementedError
    searcher = SearcherClass(policy=policy, 
                            preprocess_obs_func=preprocess_obs_func, 
                            k=k, device=device,
                            mask_actions=mask_actions, 
                            maskable_policy=maskable_policy)
    return searcher


class Searcher:
    
    def __init__(self, policy, preprocess_obs_func, k=1, device=None, mask_actions=True, maskable_policy=True, allow_parallel=True):
        self.policy = policy
        self.preprocess_obs_func = preprocess_obs_func
        self.k = k
        self.allow_parallel = allow_parallel
        self.allow_parallel = False if k == 1 else allow_parallel
        self.mask_actions = mask_actions
        self.maskable_policy = maskable_policy
        self.mp_pool = None
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.softmax_temp = 1. + np.log2(k) * 0.01

    def set_mp_pool(self):
        assert self.allow_parallel
        if self.mp_pool is not None: self.mp_pool.close()
        num_processes =  min(multiprocessing.cpu_count(), self.k)
        self.mp_pool = mp.Pool(processes=num_processes, maxtasksperchild=num_processes * 100)

    
    def find_solution(self, SubEnv):
        raise NotImplementedError

    def get_action_distribution(self, observation, mask=None):
        with torch.no_grad():
            action_logits = self.policy.act(observation)
        if mask is not None and self.mask_actions:
            candicate_action_logits = apply_mask_to_logit(action_logits, mask)
        else:
            candicate_action_logits = action_logits
        candicate_action_probs = F.softmax(candicate_action_logits / self.softmax_temp, dim=-1)
        return candicate_action_probs

    def select_action(self, observation, mask=None, sample=True):
        with torch.no_grad():
            action_logits = self.policy.act(observation)

        if mask is not None and self.mask_actions:
            candicate_action_logits = apply_mask_to_logit(action_logits, mask) 
        else:
            candicate_action_logits = action_logits

        if self.mask_actions and self.maskable_policy:
            candicate_action_probs = F.softmax(candicate_action_logits / self.softmax_temp, dim=-1)
            candicate_action_dist = Categorical(probs=candicate_action_probs)
        else:
            candicate_action_probs = F.softmax(action_logits / self.softmax_temp, dim=-1)
            candicate_action_dist = Categorical(probs=candicate_action_dist)

        if sample:
            action = candicate_action_dist.sample()
        else:
            action = candicate_action_logits.argmax(-1)

        action_logprob = candicate_action_dist.log_prob(action)
        # action = action.squeeze(-1).cpu()
        return action, action_logprob


class RandomSearcher(Searcher):

    def find_solution(self, sub_env):
        done = False
        while not done:
            mask = sub_env.generate_action_mask()
            candicate_actions = [action for action, mask in enumerate(mask) if mask]
            action = random.choice(candicate_actions)

            obs, reward, done, info = sub_env.step(action)

            if done:
                return sub_env.solution
            
        raise Exception('')


class GreedySearcher(Searcher):

    def find_solution(self, sub_env):
        obs = sub_env.get_observation()
        done = False
        while not done:
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            tensor_obs_list = self.preprocess_obs_func(obs, device=self.device)
            action, action_logprob = self.select_action(tensor_obs_list, mask=mask, sample=False)

            obs, reward, done, info = sub_env.step(action[0])

            if done:
                return sub_env.solution
            
        raise Exception('')


class SampleSearcher(Searcher):
    
    def __init__(self, policy, preprocess_obs_func, k, device=None, mask_actions=True, maskable_policy=True, allow_parallel=True):
        super(SampleSearcher, self).__init__(policy, preprocess_obs_func, k, device, mask_actions, maskable_policy)

    def find_solution(self, sub_env):
        # children = mp.active_children()
        # print(len(children))
        if self.allow_parallel: self.set_mp_pool()

        env_list = [copy.deepcopy(sub_env) for i in range(self.k)]
        obs_list = [env.get_observation() for env in env_list]
        done_list = [False] * self.k
        while not sum(done_list):
            mask = np.array([env.generate_action_mask() for env in env_list])
            tensor_obs_list = self.preprocess_obs_func(obs_list, device=self.device)
            actions, action_logprobs = self.select_action(tensor_obs_list, mask=mask, sample=True)

            t1 = time.time()
            if self.allow_parallel:
                need_stepped_env_id_list = [i for i in range(self.k) if not done_list[i]]
                need_stepped_env_list = [env_list[i] for i in range(self.k) if not done_list[i]]
                need_stepped_action_list = [int(actions[i]) for i in range(self.k) if not done_list[i]]
                results = self.mp_pool.map(env_step, list(zip(need_stepped_env_list, need_stepped_action_list)))
                for i, result in enumerate(results):
                    env_id = need_stepped_env_id_list[i]
                    env, (obs, reward, done, info) = result
                    env_list[env_id] = env
                    obs_list[env_id] = obs
                    done_list[env_id] = done
            else:
                for i, env in enumerate(env_list):
                    # continue do
                    if not done_list[i]:
                        obs, reward, done, info = env.step(actions[i])
                        obs_list[i] = obs
                        done_list[i] = done
            t2 = time.time()

            if sum(done_list) == self.k:
                break

        score_list = [env.solution.v_net_r2c_ratio if env.solution.result else 0. for env in env_list]
        solution_list = [str(env.solution['node_slots']) for env in env_list]
        # print(env_list[0].v_net.num_nodes, score_list)
        best_index = score_list.index(max(score_list))
        return env_list[best_index].solution


class BeamSearcher(Searcher):
    
    def __init__(self, policy, preprocess_obs_func, k, device=None, mask_actions=True, maskable_policy=True, allow_parallel=True):
        super(BeamSearcher, self).__init__(policy, preprocess_obs_func, k, device, mask_actions, maskable_policy)

    def find_solution(self, sub_env):
        if self.allow_parallel: self.set_mp_pool()
        
        env_list = [copy.deepcopy(sub_env) for i in range(self.k)]
        obs_list = [env.get_observation() for env in env_list]
        done_list = [False] * self.k
        global_conditional_prob_list = [1.] * self.k
        first_flag = True
        while not sum(done_list):
            t1 = time.time()
            mask = np.array([env.generate_action_mask() for env in env_list])
            tensor_obs_list = self.preprocess_obs_func(obs_list, device=self.device)
            candicate_action_probs_list = self.get_action_distribution(tensor_obs_list, mask)

            # update selected conditional probs
            current_step_prob_dict = {}
            for env_id in range(self.k):
                probs, indices = torch.topk(candicate_action_probs_list[env_id], self.k)
                for prob_id in range(self.k):
                    current_step_prob_dict[(env_id, int(indices[prob_id]))] = global_conditional_prob_list[env_id] * probs[prob_id] * (not done_list[env_id])
            if first_flag:
                for e_p, prob in current_step_prob_dict.items():
                    if e_p[0] != 0:
                        current_step_prob_dict[e_p] *= 0
                first_flag = False
            
            # select top-k (env_id, action)
            sorted_list = list(sorted(current_step_prob_dict.items(), key=lambda item: item[1], reverse=True))
            topk_probs = sorted_list[:self.k]
            env_id_list = [topk_probs[i][0][0] for i in range(self.k)]
            env_list = [copy.deepcopy(env_list[topk_probs[i][0][0]]) for i in range(self.k)]
            actions = [topk_probs[i][0][1] for i in range(self.k)]
            global_conditional_prob_list = [topk_probs[i][1] for i in range(self.k)]
            
            t1 = time.time()
            if self.allow_parallel:
                need_stepped_env_id_list = [i for i in range(self.k) if not done_list[i]]
                need_stepped_env_list = [env_list[i] for i in range(self.k) if not done_list[i]]
                need_stepped_action_list = [actions[i] for i in range(self.k) if not done_list[i]]
                results = self.mp_pool.map(env_step, list(zip(need_stepped_env_list, need_stepped_action_list)))
                for i, result in enumerate(results):
                    env_id = need_stepped_env_id_list[i]
                    env, (obs, reward, done, info) = result
                    env_list[env_id] = env
                    obs_list[env_id] = obs
                    done_list[env_id] = done
            else:
                for i, env in enumerate(env_list):
                    # continue do
                    if not done_list[i]:
                        obs, reward, done, info = env.step(actions[i])
                        obs_list[i] = obs
                        done_list[i] = done
            t2 = time.time()
            # print(t2- t1)
            
            if sum(done_list) == self.k:
                break
        score_list = [env.solution.v_net_r2c_ratio if env.solution.result and env.solution.violation <= 0 else 0. for env in env_list]
        # violation_list = [env.solution.violation for env in env_list]
        solution_list = [str(list(env.solution['node_slots'].items())) for env in env_list]
        
        # print(score_list)
        # print(global_conditional_prob_list)
        best_index = score_list.index(max(score_list))
        greedy_index = global_conditional_prob_list.index(max(global_conditional_prob_list))
        print(f'num_solutions: {len(set(solution_list))}, \
                num_scores: {len(set(score_list))}, \
                best_score: {score_list[best_index]:.4f}, \
                best_index: {best_index}, \
                greedy_index: {greedy_index}')

        best_solution = env_list[best_index].solution
        best_solution.num_feasible_solutions = sum(1 if s != 0 else 0 for s in score_list)
        best_solution.num_various_solutions = len(set(score_list))
        best_solution.best_solution_index = best_index
        best_solution.best_solution_prob = float(global_conditional_prob_list[best_index])
        best_solution.best_solution_score = float(score_list[best_index])
        best_solution.greedy_solution_prob  = float(global_conditional_prob_list[greedy_index])
        best_solution.greedy_solution_score = float(score_list[greedy_index])
        return best_solution


class RecovableSearcher(Searcher):
    
    def __init__(self, policy, preprocess_obs_func, k, device=None, mask_actions=True, maskable_policy=True, allow_parallel=True):
        super(RecovableSearcher, self).__init__(policy, preprocess_obs_func, k, device, mask_actions, maskable_policy)
        # if allow_parallel:
        #     self.mp_pool = mp.Pool(processes= min(multiprocessing.cpu_count(), k))

    def find_solution(self, sub_env):
        obs = sub_env.get_observation()
        done = False
        max_retry_times = self.k
        # every_v_node_retry_times = [int(np.power(max_retry_times, 1 / (sub_env.v_net.num_nodes - i))) for i in range(sub_env.v_net.num_nodes)]
        each_v_node_retry_times = int(max_retry_times / sub_env.v_net.num_nodes)
        num_retry_times = 0
        failed_actions_dict = defaultdict(list)
        
        while not done:
            backup_sub_env = copy.deepcopy(sub_env)
            mask = sub_env.generate_action_mask()
            mask[failed_actions_dict[str(sub_env.solution.node_slots), sub_env.curr_v_node_id]] = False
            mask = np.expand_dims(mask, axis=0)
            tensor_obs_list = self.preprocess_obs_func(obs, device=self.device)
            action, action_logprob = self.select_action(tensor_obs_list, mask=mask, sample=False)
            obs, reward, done, info = sub_env.step(action[0])

            if done:
                # SUCCESS
                if sub_env.solution['result']:
                    sub_env.solution['num_retry_times'] = num_retry_times
                    return sub_env.solution
                else:
                    # FAILURE
                    if num_retry_times == max_retry_times:
                        sub_env.solution['num_retry_times'] = num_retry_times
                        return sub_env.solution
                    # Retry Current V Node
                    if num_retry_times < each_v_node_retry_times:
                        sub_env = copy.deepcopy(backup_sub_env)
                        failed_actions_dict[str(sub_env.solution.node_slots), sub_env.curr_v_node_id].append(int(action[0]))
                        obs = sub_env.get_observation()
                        done = False
                        num_retry_times += 1
                    # Retry Last V Node
                    else:
                        sub_env = copy.deepcopy(backup_sub_env)
                        if len(sub_env.solution.node_slots):
                            last_v_node_id = sub_env.placed_v_net_nodes[-1]
                            paired_p_node_id = sub_env.selected_p_net_nodes[-1]
                            sub_env.revoke()
                            failed_actions_dict[str(sub_env.solution.node_slots), last_v_node_id].append(paired_p_node_id)
                        else:
                            failed_actions_dict[str(sub_env.solution.node_slots), sub_env.curr_v_node_id].append(paired_p_node_id)
                        done = False
                        num_retry_times += 1


def env_step(env_action):
    t1 = time.time()
    env, action = env_action
    res = env.step(action)
    t2 = time.time()
    # print(os.getppid(), os.getpid(), env, action, t2-t1)
    return env, res


class OneShotSearcher:
    
    def __init__(self, policy, preprocess_obs_func, k=1, device=None, mask_actions=True, maskable_policy=True, allow_parallel=True):
        self.policy = policy
        self.preprocess_obs_func = preprocess_obs_func
        self.k = k
        self.allow_parallel = allow_parallel
        self.mask_actions = mask_actions
        self.maskable_policy = maskable_policy
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def find_one_solution(self, sub_env):
        heatmap = np.random.random(sub_env.v_net.num_nodes, sub_env.p_net.num_nodes)
        node_slots = {}
        for i in range(sub_env.v_net.num_nodes):
            if heatmap.sum() == 0:
                break
            v_id, p_id = np.unravel_index(heatmap.argmax(), heatmap.shape)
            print(v_id, p_id)
            node_slots[v_id] = p_id
            heatmap[v_id][:] = 0.
