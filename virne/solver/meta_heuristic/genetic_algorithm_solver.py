# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from concurrent.futures import wait
import copy
import random
import threading
import numpy as np

from virne.core.environment import SolutionStepEnvironment
from virne.network import VirtualNetwork, PhysicalNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.solver.base_solver import Solver, SolverRegistry


from .base_meta_heuristic_solver import Individual, BaseMetaHeuristicSolver
from ..rank.node_rank import rank_nodes


class Chromosome(Individual):
    """
    Chromosome for Genetic Algorithm Solver
    """
    def __init__(self, id, v_net, p_net):
        super(Chromosome, self).__init__(id, v_net, p_net)


@SolverRegistry.register(solver_name='ga_meta', solver_type='meta_heuristic')
class GeneticAlgorithmSolver(BaseMetaHeuristicSolver):
    """
    Genetic Algorithm (GA) Solver for VNE

    References:
        - Peiying Zhang et al. "Virtual network embedding based on modified genetic algorithm". In Peer-to-Peer Networking and Applications, 2019.
    
    Attributes:
        num_environments: number of environments
        num_chromosomes: number of chromosomes
        max_iteration: max iteration
        prob_crossover: crossover probablity
        prob_mutation: mutation probablity
        duplication_method: duplication method
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(GeneticAlgorithmSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        # super parameters
        self.num_environments = 1
        self.num_chromosomes = 8   # number of chromosomes
        self.max_iteration = 12    # max iteration
        self.prob_crossover = 0.8  # crossover probablity
        self.duplication_method = 'roulette_wheel'
        assert self.num_chromosomes != 0 and self.num_chromosomes % 2 == 0

    def ready(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        num_v_nodes = self.v_net.num_nodes
        self.prob_mutation = 1 / (num_v_nodes * self.num_chromosomes / 2) # mutation probablity
        rank_nodes(p_net, method='order')

    def initialize(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        # chromosomes
        self.chromosomes = [Chromosome(i, v_net, p_net) for i in range(self.num_chromosomes)]
        # initialize chromosomes best experience
        for chromosome in self.chromosomes:
            node_slots = self.generate_initial_node_slots(v_net, p_net, select_method='random')
            chromosome.update_solution(node_slots=node_slots)
            self.controller.deploy_with_node_slots(v_net, p_net, node_slots, chromosome.solution, inplace=False, shortest_method=self.shortest_method, k_shortest=self.k_shortest)
            self.counter.count_solution(v_net, chromosome.solution)
            chromosome.best_solution = copy.deepcopy(chromosome.solution)
        # initialize global best experience
        self.update_best_individual(self.chromosomes)

    def meta_run(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        temp_dict = {}
        unnecessary_attributes = ['policy', 'optimizer', 'lr_scheduler', 'searcher', 'writer', 'logger']
        for attr_name in unnecessary_attributes:
            if hasattr(self, attr_name):
                temp_dict[attr_name] = getattr(self, attr_name)
                delattr(self, attr_name)
        solver_list = [copy.deepcopy(self) for i in range(self.num_environments)]
        # restore attributes
        for attr_name in temp_dict.keys():
            setattr(self, attr_name, temp_dict[attr_name])
        
        mt_pool = self.get_mt_pool(self.num_environments)
        futures = [mt_pool.submit(single_meta_run, solver_list[i], v_net, p_net) for i in range(self.num_environments)]
        wait(futures)

        run_results = []
        for f in futures:
            try:
                solution, fitness = f.result()
                if solution is not None and fitness is not None:
                    run_results.append({'solution': solution, 'fitness': fitness})
                elif solution is not None: # Solution exists but fitness is None
                    self.logger.debug(f"Meta run produced a solution with None fitness, treating as worst: {solution}")
                    run_results.append({'solution': solution, 'fitness': float('inf')})
            except Exception as e:
                self.logger.error(f"A meta_run subprocess raised an exception: {e}")
        
        if not run_results:
            self.logger.warning("No successful results from any meta-heuristic environment.")
            # Create a new Solution object, assuming v_net is available in this scope
            return Solution(v_net=v_net) # Adjusted to correct Solution constructor

        best_run = min(run_results, key=lambda x: x['fitness'] if x['fitness'] is not None else float('inf'))
        
        if best_run['fitness'] == float('inf') or best_run['fitness'] is None:
            self.logger.warning("All meta-heuristic runs resulted in infeasible or failed solutions.")
            return Solution(v_net=v_net) # Adjusted to correct Solution constructor

        self.logger.debug(f"Best fitness found among environments: {best_run['fitness']}")
        return best_run['solution']

    # def meta_run(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
    #     # initialization
    #     self.initialize(v_net, p_net)
    #     # start iterating
    #     for iter_id in range(self.max_iteration):
    #         # selection
    #         self.selection()
    #         # crossover
    #         self.crossover()
    #         # mutation
    #         self.mutation()
    #         # update best
    #         self.conclusion()
    #         # update best individual
    #         self.update_best_individual(self.chromosomes)
    #         # recording
    #     return self.best_individual.best_solution

    def selection(self, k=3):
        self.selected_individuals = []
        fitness_list = [chromosome.fitness for chromosome in self.chromosomes]
        # Ensure all fitness values are not None before calculating selection_probs
        if any(f is None for f in fitness_list):
            # Handle cases where fitness might be None, e.g., by assigning a default high value or skipping
            # For now, if any fitness is None, use uniform probability as a fallback
            self.logger.warning("None fitness detected in chromosomes during selection, using uniform probabilities.")
            selection_probs = [1/len(fitness_list) for _ in fitness_list]
        else:
            selection_probs = [1/v if v != 0 else float('inf') for v in fitness_list] # Avoid division by zero
            # Normalize probabilities
            sum_probs = sum(p for p in selection_probs if p != float('inf'))
            if sum_probs == 0:
                # If all inverse fitnesses are inf (i.e., all fitnesses were 0) or sum is 0 for other reasons
                selection_probs = [1/len(fitness_list) for _ in fitness_list]
            else:
                selection_probs = [p/sum_probs if p != float('inf') else 0 for p in selection_probs]
                # Check if sum is still zero (e.g. all probs were inf)
                if sum(selection_probs) == 0:
                     selection_probs = [1/len(fitness_list) for _ in fitness_list]

        # selection
        if self.duplication_method == 'tournament':
            for i in range(self.num_chromosomes):
                sub_population = random.choices(self.chromosomes, selection_probs, k=k)
                sub_population_fitness = [individual.fitness for individual in sub_population]
                selected_individual_index = sub_population_fitness.index(min(sub_population_fitness))
                self.selected_individuals.append(copy.deepcopy(sub_population[selected_individual_index]))
        if self.duplication_method == 'roulette_wheel':
            selected_indices = random.choices(list(range(0, self.num_chromosomes)), selection_probs, k=self.num_chromosomes)
            self.selected_individuals = [copy.deepcopy(self.chromosomes[ind_id]) for ind_id in selected_indices]
        return self.selected_individuals

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
                    # Ensure solutions are not None before accessing
                    if ind_1.solution and ind_1.solution['node_slots'] and \
                       ind_2.solution and ind_2.solution['node_slots'] and \
                       crossover_ind_1.solution and crossover_ind_1.solution['node_slots'] and \
                       crossover_ind_2.solution and crossover_ind_2.solution['node_slots']:
                        crossover_ind_1.solution['node_slots'][v_node_id] = ind_2.solution['node_slots'][v_node_id]
                        crossover_ind_2.solution['node_slots'][v_node_id] = ind_1.solution['node_slots'][v_node_id]
                    else:
                        # Handle cases where solution or node_slots might be None
                        # This might involve skipping crossover for this pair or re-initializing
                        self.logger.warning("Skipping crossover due to None solution/node_slots in individuals.")
                        self.next_generation.append(ind_1)
                        self.next_generation.append(ind_2)
                        continue # Skip to next pair

                # repair infeasible solution using the same p_node repeatly
                for crossover_ind, ind in zip([crossover_ind_1, crossover_ind_2], [ind_1, ind_2]):
                    if crossover_ind.solution and crossover_ind.solution['node_slots'] and \
                       len(set(crossover_ind.solution['node_slots'].values())) < self.v_net.num_nodes:
                        repair_result = self.repair(crossover_ind, fixed_v_node_id_list=crossover_v_node_id_list)
                        if repair_result:
                            self.next_generation.append(crossover_ind)
                        else:
                            self.next_generation.append(ind)
                    elif crossover_ind.solution and crossover_ind.solution['node_slots']:
                        self.next_generation.append(crossover_ind)
                    else:
                        # If crossover_ind.solution or node_slots is None, append original individual
                        self.next_generation.append(ind)
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
            if c.solution and c.solution['node_slots']:
                self.controller.deploy_with_node_slots(c.v_net, c.p_net, c.solution['node_slots'], c.solution, inplace=False, shortest_method=self.shortest_method, k_shortest=self.k_shortest)
                self.counter.count_solution(c.v_net, c.solution)
                c.update_best_solution()
            else:
                self.logger.warning(f"Chromosome {c.id} has no solution or node_slots, skipping conclusion step.")

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


def single_meta_run(solver, v_net, p_net):
    """
    Run a single meta-heuristic solver
    """
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
    
    if solver.best_individual is not None:
        fitness = solver.best_individual.fitness
        solution = solver.best_individual.solution
        if solution is None:
            # If solution is None, this is a critical issue, treat as worst fitness.
            # Log this occurrence if a logger is available and serializable.
            return None, float('inf') 
        if fitness is None:
            # This case implies that the objective was not set or is None.
            # Treat as worst-case fitness.
            return solution, float('inf')
        return solution, fitness
    else:
        # This solver instance failed to find any best_individual.
        return None, float('inf')