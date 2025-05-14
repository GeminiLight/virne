# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================

import copy
import random
import threading
import numpy as np

from virne.core.environment import SolutionStepEnvironment
from virne.solver.meta_heuristic.base_meta_heuristic_solver import Individual, BaseMetaHeuristicSolver
from virne.network import VirtualNetwork, PhysicalNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.solver.base_solver import Solver, SolverRegistry


class TabuIndividual(Individual):
    def __init__(self, id, v_net, p_net):
        super(TabuIndividual, self).__init__(id, v_net, p_net)
        self.tabu_list = []
        self.last_solution = copy.deepcopy(self.solution)  # Initialize last_solution


@SolverRegistry.register(solver_name='ts_meta', solver_type='meta_heuristic')
class TabuSearchSolver(BaseMetaHeuristicSolver):
    """
    Tabu Search Algorithm for VNE

    Attributes:
        num_individuals: number of individuals
        max_iteration: max iteration
        tabu_tenure: tenure of tabu list
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(TabuSearchSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        # super parameters
        self.num_individuals = 8
        self.max_iteration = 12
        self.tabu_tenure = 5
        self.max_attempt_times = 2

    def ready(self, v_net, p_net):
        return super().ready(v_net, p_net)

    def meta_run(self, v_net, p_net):
        # initialization
        self.initialize(v_net, p_net)
        # start iterating
        individual_runners = []
        for individual in self.individuals:
            individual_runners.append(threading.Thread(target=self.evolve, args=(individual, )))
        for individual_runner in individual_runners:
            individual_runner.start()
        for individual_runner in individual_runners:
            individual_runner.join()
        self.update_best_individual(self.individuals)
        return self.best_individual.best_solution

    def initialize(self, v_net, p_net):
        # individuals
        self.individuals = [TabuIndividual(i, v_net, p_net) for i in range(self.num_individuals)]
        # initialize individuals best experience
        for individual in self.individuals:
            node_slots = self.generate_initial_node_slots(v_net, p_net, select_method='random')
            individual.update_solution(node_slots=node_slots)
            self.controller.deploy_with_node_slots(v_net, 
                                                    p_net, node_slots, 
                                                    individual.solution, 
                                                    inplace=False,
                                                    shortest_method=self.shortest_method,
                                                    k_shortest=self.k_shortest)
            self.counter.count_solution(v_net, individual.solution)
            individual.update_best_solution()
            individual.last_solution = copy.deepcopy(individual.solution)  # Initialize last_solution
        # initialize global best experience
        self.update_best_individual(self.individuals)
        return self.best_individual

    def generate_neighbor(self, individual):
        for i in range(self.max_attempt_times):
            changed_v_node_id = random.randint(0, individual.v_net.num_nodes-1)
            current_p_node = individual.solution['node_slots'][changed_v_node_id]
            p_candidate = self.select_p_candidate(changed_v_node_id, individual.selected_p_nodes, p_node_weights=None, method='random')
            if p_candidate != -1 and p_candidate != current_p_node and (changed_v_node_id, p_candidate) not in individual.tabu_list:
                individual.update_solution(node_slots={changed_v_node_id: p_candidate})
                self.controller.deploy_with_node_slots(individual.v_net, individual.p_net, individual.solution['node_slots'], individual.solution, inplace=False)
                self.counter.count_solution(individual.v_net, individual.solution)
                individual.tabu_list.append((changed_v_node_id, p_candidate))
                if len(individual.tabu_list) > self.tabu_tenure:
                    individual.tabu_list.pop(0)
                break

    def evolve(self, individual):
        iter_id = 0
        while iter_id < self.max_iteration:
            self.generate_neighbor(individual)
            last_fitness = individual.calc_fitness(individual.last_solution)
            curr_fitness = individual.calc_fitness(individual.solution)
            if curr_fitness < last_fitness:
                individual.last_solution = copy.deepcopy(individual.solution)
                individual.update_best_solution()
            else:
                individual.solution = copy.deepcopy(individual.last_solution)
            iter_id += 1
