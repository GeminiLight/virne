# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


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

from ..utils import apply_mask_to_logit


def get_searcher(decode_strategy, policy, preprocess_obs_func, k, device, mask_actions, maskable_policy, make_policy_func):
    if decode_strategy in [0, 'random']:
        SearcherClass = RandomSearcher
    elif decode_strategy in [1, 'greedy']:
        SearcherClass = GreedySearcher
    elif decode_strategy in [1, 'sample', 'sampling'] and k == 1:
        SearcherClass = SingleSampleSearcher
    elif decode_strategy in [1, 'sample', 'sampling'] and k != 1:
        SearcherClass = SampleSearcher
    elif decode_strategy in [2, 'beam', 'beam_search']:
        SearcherClass = BeamSearcher
    elif decode_strategy in [3, 'recoverable', 'recovable']:
        SearcherClass = RecoverableSearcher
    else:
        raise NotImplementedError
    searcher = SearcherClass(policy=policy, 
                            preprocess_obs_func=preprocess_obs_func, 
                            make_policy_func=make_policy_func,
                            k=k, device=device,
                            mask_actions=mask_actions, 
                            maskable_policy=maskable_policy)
    return searcher


def select_action(
        policy, 
        observation, 
        mask=None, 
        sample=True, 
        softmax_temp=1.0, 
        mask_actions=True, 
        maskable_policy=True):
    with torch.no_grad():
        action_logits = policy.act(observation)

    if mask is not None and mask_actions:
        candidate_action_logits = apply_mask_to_logit(action_logits, mask) 
    else:
        candidate_action_logits = action_logits

    if mask_actions and maskable_policy:
        candidate_action_probs = F.softmax(candidate_action_logits / softmax_temp, dim=-1)
        candidate_action_dist = Categorical(probs=candidate_action_probs)
    else:
        candidate_action_probs = F.softmax(action_logits / softmax_temp, dim=-1)
        candidate_action_dist = Categorical(probs=candidate_action_probs)

    if sample:
        action = candidate_action_dist.sample()
    else:
        action = candidate_action_logits.argmax(-1)

    action_logprob = candidate_action_dist.log_prob(action)

    if torch.numel(action) == 1:
        action = action.item()
    else:
        action = action.reshape(-1, ).cpu().detach().numpy()
    # action = action.squeeze(-1).cpu()
    return action, action_logprob


def _step_instance_env(instance_env, action):
    observation, reward, terminated, truncated, info = instance_env.step(action)
    return observation, reward, terminated or truncated, info

def greedy_search_solution(
        policy, 
        instance_env, 
        preprocess_obs_func, 
        device, 
        softmax_temp=1.0, 
        mask_actions=True, 
        maskable_policy=True,
    ):
    obs = instance_env.get_observation()
    done = False
    while not done:
        mask = np.expand_dims(instance_env.generate_action_mask(), axis=0)
        tensor_obs = preprocess_obs_func(obs, device=device)
        action, action_logprob = select_action(
            policy, tensor_obs, mask=mask, sample=False, softmax_temp=softmax_temp, mask_actions=mask_actions, maskable_policy=maskable_policy)
        obs, reward, done, info = _step_instance_env(instance_env, action)
        if done:
            return instance_env.solution
    raise Exception('')

def sample_search_solution(
        policy, 
        instance_env, 
        preprocess_obs_func, 
        device, 
        softmax_temp=1.0, 
        mask_actions=True, 
        maskable_policy=True
    ):
    obs = instance_env.get_observation()
    done = False
    while not done:
        mask = np.expand_dims(instance_env.generate_action_mask(), axis=0)
        tensor_obs = preprocess_obs_func(obs, device=device)
        action, action_logprob = select_action(
            policy, tensor_obs, mask=mask, sample=True, softmax_temp=softmax_temp, mask_actions=mask_actions, maskable_policy=maskable_policy)
        obs, reward, done, info = _step_instance_env(instance_env, action)
        if done:
            return instance_env.solution
    raise Exception('')

def random_search_solution(
        policy, 
        instance_env, 
        **kwargs
    ):
    done = False
    while not done:
        mask = instance_env.generate_action_mask()
        candidate_actions = [action for action, mask in enumerate(mask) if mask]
        action = random.choice(candidate_actions)

        obs, reward, done, info = _step_instance_env(instance_env, action)

        if done:
            return instance_env.solution
        
    raise Exception('')

def get_action_distribution(policy, observation, softmax_temp=1.0, mask=None, mask_actions=True):
    with torch.no_grad():
        action_logits = policy.act(observation)
    if mask is not None and mask_actions:
        candidate_action_logits = apply_mask_to_logit(action_logits, mask)
    else:
        candidate_action_logits = action_logits
    candidate_action_probs = F.softmax(candidate_action_logits / softmax_temp, dim=-1)
    return candidate_action_probs


class Searcher:
    
    def __init__(self, policy, preprocess_obs_func, make_policy_func, k=1, device=None, mask_actions=True, maskable_policy=True, parallel_searching=True):
        self.policy = policy
        self.preprocess_obs_func = preprocess_obs_func
        self.make_policy_func = make_policy_func
        self.k = k
        self.parallel_searching = parallel_searching
        self.parallel_searching = False if k == 1 else parallel_searching
        self.mask_actions = mask_actions
        self.maskable_policy = maskable_policy
        self.mp_pool = None
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.softmax_temp = 1. + np.log2(k) * 0.01

    def set_mp_pool(self):
        assert self.parallel_searching
        if self.mp_pool is not None: self.mp_pool.close()
        num_processes =  min(multiprocessing.cpu_count(), self.k)
        self.mp_pool = mp.Pool(processes=num_processes, maxtasksperchild=num_processes * 100)
    
    def find_solution(self, InstanceEnv):
        raise NotImplementedError

    def select_action(self, observation, sample=True):
        with torch.no_grad():
            action_logits = self.policy.act(observation)

        if 'action_mask' in observation and self.mask_actions:
            mask = observation['action_mask']
            candidate_action_logits = apply_mask_to_logit(action_logits, mask) 
        else:
            candidate_action_logits = action_logits

        if self.mask_actions and self.maskable_policy:
            candidate_action_probs = F.softmax(candidate_action_logits / self.softmax_temp, dim=-1)
            candidate_action_dist = Categorical(probs=candidate_action_probs)
        else:
            candidate_action_probs = F.softmax(action_logits / self.softmax_temp, dim=-1)
            candidate_action_dist = Categorical(probs=candidate_action_probs)

        if sample:
            action = candidate_action_dist.sample()
        else:
            action = candidate_action_logits.argmax(-1)

        action_logprob = candidate_action_dist.log_prob(action)

        if torch.numel(action) == 1:
            action = action.item()
        else:
            action = action.reshape(-1, ).cpu().detach().numpy()
        # action = action.squeeze(-1).cpu()
        return action, action_logprob


class RandomSearcher(Searcher):

    def find_solution(self, instance_env):
        done = False
        while not done:
            mask = instance_env.generate_action_mask()
            candidate_actions = [action for action, mask in enumerate(mask) if mask]
            action = random.choice(candidate_actions)

            obs, reward, done, info = _step_instance_env(instance_env, action)

            if done:
                return instance_env.solution
            
        raise Exception('')


class GreedySearcher(Searcher):

    def find_solution(self, instance_env):
        return greedy_search_solution(self.policy, instance_env, self.preprocess_obs_func, self.device, 
                                   softmax_temp=self.softmax_temp, mask_actions=self.mask_actions, maskable_policy=self.maskable_policy)


class SingleSampleSearcher(Searcher):

    def find_solution(self, instance_env):
        return sample_search_solution(self.policy, instance_env, self.preprocess_obs_func, self.device, 
                                   softmax_temp=self.softmax_temp, mask_actions=self.mask_actions, maskable_policy=self.maskable_policy)



class GreedyWithRestartSearcher(Searcher):

    def find_solution(self, instance_env):
        obs = instance_env.get_observation()
        done = False
        while not done:
            tensor_obs =self.preprocess_obs_func(obs, device=self.device)
            mask = np.expand_dims(instance_env.generate_action_mask(), axis=0)
            with torch.no_grad():
                action_logits = self.policy.act(tensor_obs)
            if 'action_mask' in obs and self.mask_actions:
                mask = obs['action_mask']
                candidate_action_logits = apply_mask_to_logit(action_logits, mask)
            else:
                candidate_action_logits = action_logits
            candidate_action_probs = F.softmax(candidate_action_logits / self.softmax_temp, dim=-1)
            # get action id order by prob
            candidate_action_probs = candidate_action_probs.cpu().detach().numpy()
            action_id_order = np.argsort(-candidate_action_probs).reshape(-1, )
            action_id_order = action_id_order
            
            for i in range(2):
                obs, reward, done, info = _step_instance_env(
                    instance_env,
                    action_id_order[i],
                )

                if done and not instance_env.solution['result']:
                    pass

                if done and instance_env.solution['result']:
                    return instance_env.solution
        raise Exception('')


        return greedy_search_solution(self.policy, instance_env, self.preprocess_obs_func, self.device, 
                                   softmax_temp=self.softmax_temp, mask_actions=self.mask_actions, maskable_policy=self.maskable_policy)






class SampleSearcher(Searcher):

    def __init__(self, policy, preprocess_obs_func, make_policy_func, k, device=None, mask_actions=True, maskable_policy=True, parallel_searching=True):
        super(SampleSearcher, self).__init__(policy, preprocess_obs_func, make_policy_func, k, device, mask_actions, maskable_policy, parallel_searching)
        if self.device.type != 'cpu':
            self.parallel_searching = False
        if self.parallel_searching:
            self.policy.share_memory()
        # self.policy_list = [copy.deepcopy(policy).to('cuda') for i in range(k)]
        # self.device = torch.device('cpu')

    def find_solution(self, instance_env):
        if self.k == 1:
            return sample_search_solution(self.policy, instance_env, self.preprocess_obs_func, self.device, 
                                       softmax_temp=self.softmax_temp, mask_actions=self.mask_actions, maskable_policy=self.maskable_policy)

        # exclude the "logger"and restore it after
        logger = instance_env.logger
        instance_env.logger = None
        instance_env_list = [copy.deepcopy(instance_env) for i in range(self.k)]

        num_processes = min(multiprocessing.cpu_count(), self.k)
        args_list = [(
            self.policy,
            instance_env_list[i],
            self.preprocess_obs_func,
            self.device,
            self.softmax_temp,
            self.mask_actions,
            self.maskable_policy,
        ) for i in range(self.k)]
        try:
            if self.parallel_searching:
                with mp.Pool(
                    processes=num_processes,
                    maxtasksperchild=num_processes * 100,
                ) as mp_pool:
                    solutions = mp_pool.starmap(sample_search_solution, args_list)
            else:
                solutions = [sample_search_solution(*args) for args in args_list]
        finally:
            instance_env.logger = logger
        # solutions = []
        # for i in range(self.k):
        #     solution = search_one_solution(self.policy_list[0], instance_env_list[i], self.preprocess_obs_func, self.device, sample=False, 
        #                                    softmax_temp=self.softmax_temp, mask_actions=self.mask_actions, maskable_policy=self.maskable_policy)
        #     solutions.append(solution)

        score_list = [solution.v_net_r2c_ratio if solution.result else 0. for solution in solutions]
        best_index = score_list.index(max(score_list))
        return solutions[best_index]


class BeamSearcher(Searcher):
    
    def __init__(self, policy, preprocess_obs_func, make_policy_func, k, device=None, mask_actions=True, maskable_policy=True, parallel_searching=True):
        super(BeamSearcher, self).__init__(
            policy,
            preprocess_obs_func,
            make_policy_func,
            k,
            device,
            mask_actions,
            maskable_policy,
            parallel_searching,
        )

    def find_solution(self, instance_env):
        # Each item is (environment, observation, cumulative log probability,
        # done). Completed beams are retained while live beams are expanded.
        # Keeping these fields together prevents parent/observation/done state
        # from drifting apart after branching.
        initial_env = copy.deepcopy(instance_env)
        beams = [(initial_env, initial_env.get_observation(), 0.0, False)]
        max_steps = max(1, 10 * instance_env.v_net.num_nodes + 1)

        for _ in range(max_steps):
            expanded_beams = []
            for env, observation, cumulative_log_prob, done in beams:
                if done:
                    expanded_beams.append(
                        (env, observation, cumulative_log_prob, True)
                    )
                    continue

                mask = env.generate_action_mask()
                feasible_actions = (
                    np.flatnonzero(mask)
                    if self.mask_actions
                    else np.arange(env.num_actions)
                )
                if feasible_actions.size == 0:
                    raise RuntimeError('Beam search received an empty action mask')
                tensor_obs = self.preprocess_obs_func(
                    observation,
                    device=self.device,
                )
                action_probs = get_action_distribution(
                    self.policy,
                    tensor_obs,
                    softmax_temp=self.softmax_temp,
                    mask=np.expand_dims(mask, axis=0),
                    mask_actions=self.mask_actions,
                ).reshape(-1)
                action_probs = action_probs.detach().cpu().numpy()
                ranked_actions = sorted(
                    feasible_actions,
                    key=lambda action: action_probs[action],
                    reverse=True,
                )[:self.k]

                for action in ranked_actions:
                    child_env = copy.deepcopy(env)
                    child_obs, _, child_done, _ = _step_instance_env(
                        child_env,
                        int(action),
                    )
                    child_log_prob = cumulative_log_prob + float(
                        np.log(max(action_probs[action], np.finfo(np.float32).tiny))
                    )
                    expanded_beams.append(
                        (child_env, child_obs, child_log_prob, child_done)
                    )

            if not expanded_beams:
                raise RuntimeError('Beam search produced no successor states')
            beams = sorted(
                expanded_beams,
                key=lambda item: item[2],
                reverse=True,
            )[:self.k]
            if all(item[3] for item in beams):
                break
        else:
            raise RuntimeError('Beam search exceeded the environment step limit')

        env_list = [item[0] for item in beams]
        global_log_prob_list = [item[2] for item in beams]
        score_list = [
            env.solution.v_net_r2c_ratio if env.solution.is_feasible() else 0.
            for env in env_list
        ]
        solution_list = [str(list(env.solution['node_slots'].items())) for env in env_list]
        best_index = score_list.index(max(score_list))
        greedy_index = global_log_prob_list.index(max(global_log_prob_list))

        best_solution = env_list[best_index].solution
        best_solution.num_feasible_solutions = sum(1 if s != 0 else 0 for s in score_list)
        best_solution.num_various_solutions = len(set(score_list))
        best_solution.best_solution_index = best_index
        best_solution.best_solution_prob = float(np.exp(global_log_prob_list[best_index]))
        best_solution.best_solution_score = float(score_list[best_index])
        best_solution.greedy_solution_prob = float(np.exp(global_log_prob_list[greedy_index]))
        best_solution.greedy_solution_score = float(score_list[greedy_index])
        return best_solution


class RecoverableSearcher(Searcher):
    
    def __init__(self, policy, preprocess_obs_func, make_policy_func, k, device=None, mask_actions=True, maskable_policy=True, parallel_searching=True):
        super(RecoverableSearcher, self).__init__(
            policy,
            preprocess_obs_func,
            make_policy_func,
            k,
            device,
            mask_actions,
            maskable_policy,
            parallel_searching,
        )
        # if parallel_searching:
        #     self.mp_pool = mp.Pool(processes= min(multiprocessing.cpu_count(), k))

    def find_solution(self, instance_env):
        obs = instance_env.get_observation()
        done = False
        max_retry_times = self.k
        # every_v_node_retry_times = [int(np.power(max_retry_times, 1 / (instance_env.v_net.num_nodes - i))) for i in range(instance_env.v_net.num_nodes)]
        each_v_node_retry_times = max(
            1,
            int(max_retry_times / instance_env.v_net.num_nodes),
        )
        num_retry_times = 0
        failed_actions_dict = defaultdict(list)
        
        while not done:
            backup_instance_env = copy.deepcopy(instance_env)
            mask = instance_env.generate_action_mask()
            mask[failed_actions_dict[str(instance_env.solution.node_slots), instance_env.curr_v_node_id]] = False
            if not mask.any():
                if len(instance_env.solution.node_slots):
                    last_v_node_id = instance_env.placed_v_net_nodes[-1]
                    paired_p_node_id = instance_env.selected_p_net_nodes[-1]
                    instance_env.revoke()
                    failed_actions_dict[
                        str(instance_env.solution.node_slots), last_v_node_id
                    ].append(instance_env.p_node_id_to_action[paired_p_node_id])
                    obs = instance_env.get_observation()
                    num_retry_times += 1
                    if num_retry_times >= max_retry_times:
                        return instance_env.solution
                    continue
                instance_env._no_feasible_action = True
                instance_env.fail_no_feasible_action()
                instance_env.solution['num_retry_times'] = num_retry_times
                return instance_env.solution
            mask = np.expand_dims(mask, axis=0)
            tensor_obs_list = self.preprocess_obs_func(obs, device=self.device)
            action, action_logprob = select_action(
                self.policy,
                tensor_obs_list,
                mask=mask,
                sample=False,
                softmax_temp=self.softmax_temp,
                mask_actions=self.mask_actions,
                maskable_policy=self.maskable_policy,
            )
            obs, reward, done, info = _step_instance_env(instance_env, action)

            if done:
                # SUCCESS
                if instance_env.solution['result']:
                    instance_env.solution['num_retry_times'] = num_retry_times
                    return instance_env.solution
                else:
                    # FAILURE
                    if num_retry_times == max_retry_times:
                        instance_env.solution['num_retry_times'] = num_retry_times
                        return instance_env.solution
                    # Retry Current V Node
                    if num_retry_times < each_v_node_retry_times:
                        instance_env = copy.deepcopy(backup_instance_env)
                        failed_actions_dict[str(instance_env.solution.node_slots), instance_env.curr_v_node_id].append(int(action))
                        obs = instance_env.get_observation()
                        done = False
                        num_retry_times += 1
                    # Retry Last V Node
                    else:
                        instance_env = copy.deepcopy(backup_instance_env)
                        if len(instance_env.solution.node_slots):
                            last_v_node_id = instance_env.placed_v_net_nodes[-1]
                            paired_p_node_id = instance_env.selected_p_net_nodes[-1]
                            instance_env.revoke()
                            failed_actions_dict[
                                str(instance_env.solution.node_slots),
                                last_v_node_id,
                            ].append(
                                instance_env.p_node_id_to_action[paired_p_node_id]
                            )
                        else:
                            failed_actions_dict[
                                str(instance_env.solution.node_slots),
                                instance_env.curr_v_node_id,
                            ].append(int(action))
                        done = False
                        num_retry_times += 1


# Keep the historical misspelling import-compatible.
RecovableSearcher = RecoverableSearcher


def env_step(env_action):
    t1 = time.time()
    env, action = env_action
    res = env.step(action)
    t2 = time.time()
    # print(os.getppid(), os.getpid(), env, action, t2-t1)
    return env, res


class OneShotSearcher:
    
    def __init__(self, policy, preprocess_obs_func, k=1, device=None, mask_actions=True, maskable_policy=True, parallel_searching=True):
        self.policy = policy
        self.preprocess_obs_func = preprocess_obs_func
        self.k = k
        self.parallel_searching = parallel_searching
        self.mask_actions = mask_actions
        self.maskable_policy = maskable_policy
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def find_one_solution(self, instance_env):
        heatmap = np.random.random(instance_env.v_net.num_nodes, instance_env.p_net.num_nodes)
        node_slots = {}
        for i in range(instance_env.v_net.num_nodes):
            if heatmap.sum() == 0:
                break
            v_id, p_id = np.unravel_index(heatmap.argmax(), heatmap.shape)
            print(v_id, p_id)
            node_slots[v_id] = p_id
            heatmap[v_id][:] = 0.





# class SampleSearcher(Searcher):
    
#     def __init__(self, policy, preprocess_obs_func, k, device=None, mask_actions=True, maskable_policy=True, parallel_searching=True):
#         super(SampleSearcher, self).__init__(policy, preprocess_obs_func, k, device, mask_actions, maskable_policy)

#     def find_solution(self, instance_env):
#         # children = mp.active_children()
#         # print(len(children))
#         if self.parallel_searching: self.set_mp_pool()

#         env_list = [copy.deepcopy(instance_env) for i in range(self.k)]
#         obs_list = [env.get_observation() for env in env_list]
#         done_list = [False] * self.k
#         while not sum(done_list):
#             mask = np.array([env.generate_action_mask() for env in env_list])
#             tensor_obs_list = self.preprocess_obs_func(obs_list, device=self.device)
#             actions, action_logprobs = self.select_action(tensor_obs_list, sample=True)
#             if self.k == 1:
#                 actions = [actions]
#                 action_logprobs = [action_logprobs]

#             t1 = time.time()
#             if self.parallel_searching:
#                 need_stepped_env_id_list = [i for i in range(self.k) if not done_list[i]]
#                 need_stepped_env_list = [env_list[i] for i in range(self.k) if not done_list[i]]
#                 need_stepped_action_list = [int(actions[i]) for i in range(self.k) if not done_list[i]]
#                 results = self.mp_pool.map(env_step, list(zip(need_stepped_env_list, need_stepped_action_list)))
#                 for i, result in enumerate(results):
#                     env_id = need_stepped_env_id_list[i]
#                     env, (obs, reward, terminated, truncated, info) = result
#                     done = terminated or truncated
#                     env_list[env_id] = env
#                     obs_list[env_id] = obs
#                     done_list[env_id] = done
#             else:
#                 for i, env in enumerate(env_list):
#                     # continue do
#                     if not done_list[i]:
#                         obs, reward, terminated, truncated, info = env.step(actions[i])
#                         done = terminated or truncated
#                         obs_list[i] = obs
#                         done_list[i] = done
#             t2 = time.time()

#             if sum(done_list) == self.k:
#                 break

#         score_list = [env.solution.v_net_r2c_ratio if env.solution.result else 0. for env in env_list]
#         solution_list = [str(env.solution['node_slots']) for env in env_list]
#         # print(env_list[0].v_net.num_nodes, score_list)
#         best_index = score_list.index(max(score_list))
#         return env_list[best_index].solution
