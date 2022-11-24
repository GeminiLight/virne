"""
Paper: Energy-Aware Virtual Network Embedding
Ref: 
    https://github.com/DiamondTan/VirtualNetworkEmbedding
    https://github.com/family9977/VNE_simulator/
"""

import copy
import random
import threading
import numpy as np

from solver.meta_heuristic.meta_heuristic_solver import Individual, MetaHeuristicSolver


class SimulatedAnnealingSolver(MetaHeuristicSolver):
    """
    Simulated Annealing
    """

    name = 'sa_vne'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(SimulatedAnnealingSolver, self).__init__(controller, recorder, counter, **kwargs)
        """
        """
        # super parameters
        self.num_individuals = 10    # number of num_individuals
        self.max_iteration = 20      # max iteration
        self.max_attempt_times = 1  # key parameters for performance
        self.initial_temperature = 2
        self.attenuation_factor = 0.95

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
        self.individuals = [Individual(i, v_net, p_net) for i in range(self.num_individuals)]
        # initialize individuals best experience
        for individual in self.individuals:
            node_slots = self.generate_initial_node_slots(v_net, p_net, select_method='random')
            individual.update_solution(node_slots=node_slots)
            self.controller.deploy_with_node_slots(v_net, p_net, node_slots, individual.solution, inplace=False)
            individual.update_best_solution()
        # initialize global best experience
        self.update_best_individual(self.individuals)
        return self.best_individual

    def generate_neighor(self, individual):
        for i in range(self.max_attempt_times):
            changed_v_node_id = random.randint(0, individual.v_net.num_nodes-1)
            current_p_node = individual.solution['node_slots'][changed_v_node_id]
            p_candicate = self.select_p_candicate(changed_v_node_id, individual.selected_p_nodes, p_node_weights=None, method='random')
            if p_candicate != -1 and p_candicate != current_p_node:
                individual.update_solution(node_slots={changed_v_node_id: p_candicate})
                self.controller.deploy_with_node_slots(individual.v_net, individual.p_net, individual.solution['node_slots'], individual.solution, inplace=False)
                break

    def evolve(self, individual):
        iter_id = 0
        temperature = self.initial_temperature
        individual.last_solution = copy.deepcopy(individual.solution)
        while iter_id < self.max_iteration:
            self.generate_neighor(individual)
            last_fitness = individual.calc_fitness(individual.last_solution)
            curr_fitness = individual.calc_fitness(individual.solution)
            diff_fitness = curr_fitness - last_fitness
            if diff_fitness < 0:
                individual.last_solution = copy.deepcopy(individual.solution)
                individual.update_best_solution()
            else:
                # acceptance criterion
                prob = np.exp(- diff_fitness / temperature)
                if random.random() < prob:
                    individual.last_solution = copy.deepcopy(individual.solution)
                else:
                    individual.solution = copy.deepcopy(individual.last_solution)
            
            iter_id += 1
            temperature *= self.attenuation_factor
