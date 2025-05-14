# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import random
import threading
import numpy as np
import multiprocessing as mp
from virne.core.environment import SolutionStepEnvironment

from virne.solver.meta_heuristic.base_meta_heuristic_solver import Individual, BaseMetaHeuristicSolver
from virne.network import VirtualNetwork, PhysicalNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.solver.base_solver import Solver, SolverRegistry


class Particle(Individual):

    def __init__(self, id, v_net, p_net):
        super(Particle, self).__init__(id, v_net, p_net)
        self.velocity = [random.randint(0, 1) for j in range(v_net.num_nodes)]

    def update_position(self, node_slots):
        self.solution.node_slots.update(node_slots)

    @property
    def position(self):
        return list(self.solution.node_slots.values())

    @property
    def best_position(self):
        return list(self.best_solution.node_slots.values())


@SolverRegistry.register(solver_name='pso_meta', solver_type='meta_heuristic')
class ParticleSwarmOptimizationSolver(BaseMetaHeuristicSolver):
    """
    Particle Swarm Optimization (PSO) Solver for VNE

    References:
        - Energy-Aware Virtual Network Embedding
        - https://github.com/DiamondTan/VirtualNetworkEmbedding
        - https://github.com/family9977/VNE_simulator/

    Attributes:
        p_i: inertia weight
        p_c: cognition weight
        p_s: social weight
        num_particles: number of num_particles
        max_iteration: max iteration
    """

    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(ParticleSwarmOptimizationSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        """
        0 < p_i < p_c < p_s < 1
        p_i + p_c + p_s = 1
        """
        # super parameters
        self.p_i = 0.1  # inertia weight
        self.p_c = 0.2  # cognition weight
        self.p_s = 0.7  # social weight
        self.num_particles = 8  # number of num_particles
        self.max_iteration = 12  # max iteration
        # discrete operations
        # sub for positions: Similar to Hamming distance
        # add for positions: Select a location based on probability
        self.sub = lambda pos_x, pos_y:[int(px == py) for px, py in zip(pos_x, pos_y)]
        self.add = lambda p1, pos_1, p2, pos_2, p3, pos_3: [np.random.choice([i,j,k],p=[p1, p2, p3]).tolist() for i, j, k in zip(pos_1, pos_2, pos_3)]

    def meta_run(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        # initialization
        self.initialize(v_net, p_net)
        # start iterating
        # mp_pool = self.get_mp_pool(self.num_particles)
        for id in range(self.max_iteration):
            # result = mp_pool.map(self.evolve, self.particles)
            particle_runners = []
            for particle in self.particles:
                particle_runners.append(mp.Process(target=self.evolve, args=(particle, )))
                # particle_runners.append(ParticleRunner(v_net, p_net, self, self.particles[i]))
            for particle_runner in particle_runners:
                particle_runner.start()
            for particle_runner in particle_runners:
                particle_runner.join()

            self.update_best_individual(self.particles)

        return self.best_individual.best_solution

    def initialize(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        # particles
        self.particles = [Particle(i, v_net, p_net) for i in range(self.num_particles)]
        # initialize particles best experience
        for particle in self.particles:
            node_slots = self.generate_initial_node_slots(v_net, p_net, select_method='random')
            particle.update_position(node_slots)
            self.controller.deploy_with_node_slots(v_net, 
                                                    p_net, node_slots, 
                                                    particle.solution, 
                                                    inplace=False,
                                                    shortest_method=self.shortest_method,
                                                    k_shortest=self.k_shortest)
            self.counter.count_solution(v_net, particle.solution)
            particle.update_best_solution()
            particle.best_solution = copy.deepcopy(particle.solution)
        # initialize global best experience
        self.update_best_individual(self.particles)

    def evolve(self, particle):
        particle.velocity = self.add(
                                    self.p_i, particle.velocity, 
                                    self.p_c, self.sub(particle.best_position, particle.position), 
                                    self.p_s, self.sub(self.best_individual.best_position, particle.position))
        for v_node_id in range(particle.v_net.num_nodes):
            if particle.velocity[v_node_id] == 0:
                node_position = self.select_p_candidate(v_node_id, particle.selected_p_nodes)
                if node_position == -1:
                    continue
                particle.update_position({v_node_id: node_position})
        self.controller.deploy_with_node_slots(particle.v_net, 
                                                particle.p_net, 
                                                particle.solution.node_slots, 
                                                particle.solution, 
                                                inplace=False,
                                                shortest_method=self.shortest_method,
                                                k_shortest=self.k_shortest)
        self.counter.count_solution(particle.v_net, particle.solution)
        particle.update_best_solution()
        