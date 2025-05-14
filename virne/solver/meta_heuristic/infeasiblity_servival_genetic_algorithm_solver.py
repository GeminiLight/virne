# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import random
import threading
import multiprocessing as mp
import numpy as np

from .genetic_algorithm_solver import GeneticAlgorithmSolver
from .base_meta_heuristic_solver import Individual, BaseMetaHeuristicSolver, INFEASIBLE_FITNESS
from ..rank.node_rank import rank_nodes
from virne.network import VirtualNetwork, PhysicalNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.solver.base_solver import Solver, SolverRegistry


class Chromosome(Individual):

    def __init__(self, id, v_net, p_net):
        super(Chromosome, self).__init__(id, v_net, p_net)

    def calc_fitness(self, solution):
        # minimize
        if solution['result'] and solution['v_net_total_hard_constraint_violation'] <= 0:
            return solution['v_net_cost'] / solution['v_net_revenue']
        # useful infeasible solution
        elif solution['result'] and solution['v_net_total_hard_constraint_violation'] > 0:
            return solution['v_net_cost'] / solution['v_net_revenue'] + solution['v_net_total_hard_constraint_violation'] / solution['v_net_revenue'] * 50
        elif not solution['result']:
            return INFEASIBLE_FITNESS

    def update_best_solution(self, ):
        if self.solution['v_net_total_hard_constraint_violation'] <= 0 and self.fitness < self.best_fitness:
            self.best_solution = copy.deepcopy(self.solution)


class ISGeneticAlgorithmSolver(GeneticAlgorithmSolver):
    """
    Genetic Algorithm

    Individual: Chromosome
    population: Chromosomes
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(ISGeneticAlgorithmSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.shortest_method = 'k_shortest'
        self.k_shortest = 10
        # super parameters
        self.num_environments = 1
        self.num_chromosomes = 10   # number of num_chromosomes
        self.max_iteration = 20    # max iteration
        self.prob_crossover = 0.8  # crossover probablity
        self.duplication_method = 'tournament'
        assert self.num_chromosomes != 0 and self.num_chromosomes % 2 == 0

    def ready(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        num_v_nodes = self.v_net.num_nodes
        self.prob_mutation = 1 / (num_v_nodes * self.num_chromosomes / 2) # mutation probablity
        rank_nodes(p_net, method='grc')

    def update_best_individual(self, population):
        if self.best_individual is None:
            self.best_individual = population[0]
        for individual in population:
            if individual.best_solution['v_net_total_hard_constraint_violation'] <= 0 and individual.best_fitness < self.best_individual.best_fitness:
                self.best_individual = individual
        return self.best_individual

    def initialize(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        # chromosomes
        self.chromosomes = [Chromosome(i, v_net, p_net) for i in range(self.num_chromosomes)]
        # initialize chromosomes best experience
        for chromosome in self.chromosomes:
            node_slots = self.generate_initial_node_slots(v_net, p_net, select_method='random')
            chromosome.update_solution(node_slots=node_slots)
            self.controller.deploy_with_node_slots(v_net, p_net, node_slots, chromosome.solution, inplace=False)
            self.counter.count_solution(v_net, chromosome.solution)
            chromosome.best_solution = copy.deepcopy(chromosome.solution)
        # initialize global best experience
        self.update_best_individual(self.chromosomes)

    def meta_run(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        solver_list = [copy.deepcopy(self) for i in range(self.num_environments)]
        runners = []
        mp_pool = self.get_mp_pool(self.num_environments)
        mp_manager = mp.Manager()
        mp_res_dict = mp_manager.dict()
        for i in range(self.num_environments):
            mp_pool.apply(single_meta_run, (i, mp_res_dict, solver_list[i], v_net, p_net)) 
        mp_pool.close()
        mp_pool.join()

        solver_best_individual_list = list(mp_res_dict.values())
        p_best_solution_fitess_list = [solver_best_individual_list[i].fitness for i in range(self.num_environments)]
        best_solution_p_id = p_best_solution_fitess_list.index(min(p_best_solution_fitess_list))
        return solver_best_individual_list[best_solution_p_id].solution

    def crossover(self):
        self.next_generation = []
        for pair_id in range(0, self.num_chromosomes, 2):
            ind_1, ind_2 = self.selected_individuals[pair_id], self.selected_individuals[pair_id+1]
            # perform crossover
            if random.random() < self.prob_crossover:
                crossover_ind_1, crossover_ind_2 = copy.deepcopy(ind_1), copy.deepcopy(ind_2) 
                v_node_id_a, v_node_id_b = np.sort(np.random.randint(0, self.v_net.num_nodes, 2))
                crossover_v_node_id_list = list(range(v_node_id_a, v_node_id_b+1))
                for v_node_id in crossover_v_node_id_list:
                    crossover_ind_1.solution['node_slots'][v_node_id] = ind_2.solution['node_slots'][v_node_id]
                    crossover_ind_2.solution['node_slots'][v_node_id] = ind_1.solution['node_slots'][v_node_id]
                # repair infeasible solution using the same p_node repeatly
                for crossover_ind, ind in zip([crossover_ind_1, crossover_ind_2], [ind_1, ind_2]):
                    if len(set(crossover_ind.solution['node_slots'].values())) < self.v_net.num_nodes:
                        repair_result = self.repair(crossover_ind, fixed_v_node_id_list=crossover_v_node_id_list)
                        if repair_result:
                            self.next_generation.append(crossover_ind)
                        else:
                            self.next_generation.append(ind)
                    else:
                        self.next_generation.append(crossover_ind)
            else:
                self.next_generation.append(ind_1)
                self.next_generation.append(ind_2)
        return self.next_generation

    def mutation(self, ):
        for individual in self.next_generation:
            for v_node_id in individual.v_net.ranked_nodes:
                if random.random() < self.prob_mutation:
                    p_candidate = self.select_p_candidate(v_node_id, 
                                                            individual.selected_p_nodes, 
                                                            p_node_weights=individual.p_net.node_ranking_values)
                    if p_candidate == -1:
                        continue
                    individual.update_solution(node_slots={v_node_id: p_candidate})
        self.chromosomes = self.next_generation
        return self.chromosomes

    def conclusion(self):
        for c in self.chromosomes:
            self.controller.deploy_with_node_slots(c.v_net, c.p_net, 
                                                            c.solution['node_slots'], 
                                                            c.solution, 
                                                            inplace=False, 
                                                            shortest_method=self.shortest_method,
                                                            k_shortest=self.k_shortest,
                                                            pruning_ratio=0.5,
                                                            if_allow_constraint_violation=True
                                                            )
            self.counter.count_solution(c.v_net, c.solution)
            c.update_best_solution()

    def repair(self, individual, fixed_v_node_id_list=[]):
        new_solution = copy.deepcopy(individual.solution)
        modifiable_v_node_id_list = set(new_solution['node_slots']).difference(set(fixed_v_node_id_list))
        fixed_p_node_id = [new_solution['node_slots'][v_node_id] for v_node_id in fixed_v_node_id_list]
        for v_node_id in modifiable_v_node_id_list:
            p_node_id = new_solution['node_slots'][v_node_id]
            # duplication p_net node
            if p_node_id in fixed_p_node_id:
                p_candidate = self.select_p_candidate(v_node_id, 
                                                        list(new_solution['node_slots'].values()), 
                                                        p_node_weights=individual.p_net.node_ranking_values)
                if p_candidate == -1:
                    return False
                new_solution['node_slots'][v_node_id] = p_candidate
            else:
                continue
        individual.solution = new_solution
        return True

def single_meta_run(i, mp_res_dict, solver, v_net, p_net):
    # initialization
    solver.initialize(v_net, p_net)
    # start iterating
    for iter_id in range(solver.max_iteration):
        # selection
        solver.selection()
        # crossover
        solver.crossover()
        # mutation
        solver.mutation()
        # update best
        solver.conclusion()
        # update best individual
        solver.update_best_individual(solver.chromosomes)
        # recording
        curr_fitnesses = solver.get_fitness_list(solver.chromosomes)
        num_infeasible_solutions = sum([1 if c.solution['v_net_total_hard_constraint_violation'] > 0 or not c.solution['result'] else 0 for c in solver.chromosomes])
        print(num_infeasible_solutions, min(curr_fitnesses), curr_fitnesses)
    mp_res_dict[i] = copy.deepcopy(solver.best_individual)
    return solver.best_individual.best_solution