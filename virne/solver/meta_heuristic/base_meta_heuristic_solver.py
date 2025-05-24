# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import random
import numpy as np
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ..base_solver import Solver
from ..rank.node_rank import rank_nodes
from virne.network import VirtualNetwork, PhysicalNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.solver.base_solver import Solver, SolverRegistry


INFEASIBLE_FITNESS = float('inf')
# INFEASIBLE_FITNESS = 100


class BaseMetaHeuristicSolver(Solver):
    """
    Meta Heuristic Solver Base Class
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(BaseMetaHeuristicSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.infeasible_fitness = INFEASIBLE_FITNESS
        self.fitness_recorder = FitnessRecorder()
        # basic methods
        self.shortest_method = kwargs.get('shortest_method', 'k_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)


    def get_parallel_executor(self, num_individuals: int, m_type: str = 'thread'):
        """
        Get parallel executor, multithread or multiprocess pool.
        
        Args:
            num_individuals: number of individuals
            m_type: thread or process
        """
        num_processes = min(num_individuals, mp.cpu_count())
        if m_type == 'thread':
            parallel_executor = ThreadPoolExecutor(num_processes)
        elif m_type == 'process':
            parallel_executor = mp.Pool(num_processes)
        else:
            return ValueError(f'Param m_type should be selected from [thread, process]: {m_type}')
        return parallel_executor

    def get_mt_pool(self, num_individuals: int) -> ThreadPoolExecutor:
        """Get a multithread pool."""
        num_processes = min(num_individuals, mp.cpu_count())
        mt_pool = ThreadPoolExecutor(num_processes)
        return mt_pool

    def get_mp_pool(self, num_individuals: int) -> mp.Pool:
        """Get a multiprocess pool."""
        num_processes = min(num_individuals, mp.cpu_count())
        mp_pool = mp.Pool(num_processes)
        return mp_pool

    def solve(self, instance: dict) -> Solution:
        """Solve the instance."""
        v_net, p_net = instance['v_net'], instance['p_net']
        self.v_net = v_net
        self.p_net = p_net
        rank_nodes(v_net, self.config.solver.node_ranking_method)
        # rank_nodes(p_net, self.config.solver.node_ranking_method)
        self.ready(v_net, p_net)
        # construct candidates
        self.candidates_dict = self.controller.construct_candidates_dict(v_net, p_net)
        self.fitness_recording = list()
        if {} in self.candidates_dict.values():
            return Solution.from_v_net(v_net)
        self.best_individual = None
        solution = self.meta_run(v_net, p_net)
        if solution['v_net_total_hard_constraint_violation'] > 0:
            return Solution.from_v_net(v_net)
        # depoly with solution
        if solution['result']:
            self.controller.deploy(v_net, p_net, solution)
        return solution

    def ready(self, v_net: VirtualNetwork, p_net: PhysicalNetwork) -> None:
        rank_nodes(p_net, method=self.config.solver.node_ranking_method)
        return

    def meta_run(self, v_net: VirtualNetwork, p_net: PhysicalNetwork) -> Solution:
        raise NotImplementedError

    def evolve(self, v_net: VirtualNetwork, p_net: PhysicalNetwork) -> Solution:
        raise NotImplementedError

    def update_best_individual(self, population: list) -> 'Individual':
        if self.best_individual is None:
            self.best_individual = population[0]
        for individual in population:
            if individual.best_fitness < self.best_individual.best_fitness:
                self.best_individual = individual
        return self.best_individual

    def reinitialize(self, individual: 'Individual') -> 'Individual':
        node_slots = self.generate_initial_node_slots(individual.v_net, individual.p_net, select_method='random')
        individual.update_position(node_slots)
        self.controller.deploy_with_node_slots(individual.v_net, individual.p_net, node_slots, individual.solution, inplace=False)
        self.counter.count_solution(individual.v_net, individual.solution)
        individual.best_solution = copy.deepcopy(individual.solution)
        return individual

    def construct_candidates_dict(self, v_net: VirtualNetwork, p_net: PhysicalNetwork) -> dict:
        self.candidates_dict = self.controller.construct_candidates_dict(v_net, p_net)
        # for v_node_id in range(v_net.num_nodes):
        #     hop_range_neighbors = nx.single_source_shortest_path_length(p_net, p_node_id, cutoff=self.hop_range)
        #     for candidate in candidates_dict[v_node_id]:
        #         if candidate not in hop_range_neighbors:
        #             candidates_dict[v_node_id].remove(candidate)
        return self.candidates_dict

    def generate_initial_node_slots(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, select_method: str = 'random') -> dict:
        node_slots = {}
        for v_node_id in v_net.ranked_nodes:
            p_candidate_id = self.select_p_candidate(v_node_id, list(node_slots.values()), 
                                                    p_node_weights=p_net.node_ranking_values, method=select_method)
            node_slots[v_node_id] = p_candidate_id
        return node_slots

    def get_p_condicates(self, v_node_id, selected_p_nodes=[]):
        if self.reusable:
            candidate_nodes = self.candidates_dict[v_node_id]
        else:
            candidate_nodes = list(set(self.candidates_dict[v_node_id]).difference(set(selected_p_nodes)))
        return candidate_nodes

    def select_p_candidate(self, v_node_id, selected_p_nodes=[], p_node_weights=None, method='random') -> int:
        # get p candidates
        p_candidates = np.array(self.get_p_condicates(v_node_id, selected_p_nodes))
        if len(p_candidates) == 0:
            return -1
        # set p node weights
        if p_node_weights is None:
            p_node_weights = np.array([1 / len(p_candidates)] * len(p_candidates))
        else:
            p_node_weights = p_node_weights[p_candidates]
        if sum(p_node_weights) <= 0:
            p_node_weights = np.array([1 / len(p_node_weights)] * len(p_node_weights))
        # set select method
        if method == 'random':
            p_candidate = random.choices(p_candidates, weights=p_node_weights, k=1)[0]
        if method == 'greedy':
            p_max_weight_index = np.argmax(p_node_weights)
            p_candidate = p_candidates[p_max_weight_index]
        return p_candidate

    def get_fitness_list(self, population: list) -> list:
        return [ind.fitness for ind in population]

    def normalize_fitnesses(self, fitness_list: list) -> list:
        fitness_list = []
        temp_list = [value for value in value_list if value !=float("inf")]
        if len(temp_list) != 0:
            max_value = np.max(temp_list)
            min_value = np.min(temp_list)
            if max_value == min_value:
                for i in range(len(value_list)):
                    if value_list[i] != float('inf'):
                        value_list[i] = 0
            else:
                value_list = [(value - min_value) / (max_value - min_value) for value in value_list]
        return value_list


class LocalSearch(BaseMetaHeuristicSolver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(LocalSearch, self).__init__(controller, recorder, counter, logger, config, **kwargs)

    def generate_neigbor(self):
        return 


class Individual:
    """
    Individual class for BaseMetaHeuristicSolver

    Minimize the objective (deployment cost)

    Attributes:
        id (int): individual id
        v_net (VirtualNetwork): virtual network
        p_net (PhysicalNetwork): physical network
        ranked_v_nodes (list): ranked v nodes
        solution (Solution): solution
        best_solution (Solution): best solution
        fitness (float): fitness
        best_fitness (float): best fitness
    """
    def __init__(self, id, v_net, p_net, ranked_v_nodes=None):
        self.id = id
        self.v_net = v_net
        self.p_net = copy.deepcopy(p_net)
        if ranked_v_nodes is None:
            self.ranked_v_nodes = list(v_net.nodes)
        else:
            self.ranked_v_nodes = ranked_v_nodes
        self.solution = Solution.from_v_net(v_net)
        for v_node_id in self.ranked_v_nodes:
            self.solution.node_slots[v_node_id] = -1
        self.best_solution = copy.deepcopy(self.solution)

    def calc_fitness(self, solution):
        # minimize
        if solution['result']:
            return solution['v_net_cost'] / solution['v_net_revenue']
            # return -solution['v_net_cost']
        return INFEASIBLE_FITNESS

    def update_solution(self, node_slots={}, link_paths={}):
        self.solution.node_slots.update(node_slots)

    def update_best_solution(self, ):
        if self.fitness < self.best_fitness:
            self.best_solution = copy.deepcopy(self.solution)

    def is_feasible(self):
        return self.solution['v_net_total_hard_constraint_violation'] <= 0

    @property
    def feasiblity(self):
        return self.solution['v_net_total_hard_constraint_violation'] <= 0

    @property
    def selected_p_nodes(self):
        return list(self.solution.node_slots.values())

    @property
    def placed_v_nodes(self):
        return list(self.solution.node_slots.keys())

    @property
    def fitness(self, ):
        return self.calc_fitness(self.solution)

    @property
    def best_fitness(self, ):
        return self.calc_fitness(self.best_solution)


class FitnessRecorder:

    def __init__(self):
        self.recordings = defaultdict(dict)
        self.global_best = defaultdict(dict)
    
    def record(self, iter_id, id_fitness_dict):
        self.recordings['iter_id'].update(id_fitness_dict)



class ParallelExecutor:

    def __init__(self, num_individuals, m_type='thread'):
        num_processes = min(num_individuals, mp.cpu_count())
        if m_type == 'thread':
            self.parallel_executor = ThreadPoolExecutor(num_processes)
        elif m_type == 'process':
            self.parallel_executor = mp.Pool(num_processes)
        else:
            return ValueError(f'Param m_type should be selected from [thread, process]: {m_type}')
        return self.parallel_executor

    def map(self, fn, *iterables, **kwargs):
        return self.parallel_executor.map(fn, *iterables, **kwargs)