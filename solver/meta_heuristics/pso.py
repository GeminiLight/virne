"""
Paper: Energy-Aware Virtual Network Embedding
Ref: 
    https://github.com/DiamondTan/VirtualNetworkEmbedding
    https://github.com/family9977/VNE_simulator/
"""

import copy
import random
import numpy as np
from threading import Thread

from base import Controller, Solution, Counter
from ..solver import Solver
from ..rank.node_rank import OrderNodeRank, GRCNodeRank, FFDNodeRank, NRMNodeRank, RWNodeRank, RandomNodeRank
from ..rank.edge_rank import OrderEdgeRank, FFDEdgeRank


class Particle(Thread):

    def __init__(self, id, vn):
        self.id = id
        self.velocity = [random.randint(0, 1) for j in range(vn.num_nodes)]
        self.position = [-1] *  vn.num_nodes
        self.solution = Solution(vn)
        # best obtained experience
        self.best_position = [-1] * vn.num_nodes
        self.best_solution = Solution(vn)

    def calc_fitness_value(self, solution):
        # minimize
        if solution['result']:
            return solution['vn_cost']
        return float('inf')

    @property
    def selected_pn_nodes(self, ):
        return list(self.solution['node_slots'].values())

    @property
    def fitness_value(self, ):
        return self.calc_fitness_value(self.solution)

    @property
    def best_fitness_value(self, ):
        return self.calc_fitness_value(self.best_solution)


class ParticleRunner(Thread):

    def __init__(self, vn, pn, pso_solver, particle):
        Thread.__init__(self)
        self.vn = vn
        self.pn = copy.deepcopy(pn)
        self.particle = particle
        self.pso_solver = pso_solver

    def run(self):
        self.particle.velocity = self.pso_solver.add(
                                    self.pso_solver.p_i, self.particle.velocity, 
                                    self.pso_solver.p_c, self.pso_solver.sub(self.particle.best_position, self.particle.position), 
                                    self.pso_solver.p_s, self.pso_solver.sub(self.pso_solver.global_best['particle'].best_position, self.particle.position))
        for v_node_id in range(self.vn.num_nodes):
            if self.particle.velocity[v_node_id] == 0:
                node_position = self.pso_solver.generate_node_position(self.particle, v_node_id)
                if node_position is None:
                    break
                self.particle.position[v_node_id] = node_position
        self.pso_solver.mapping(self.vn, self.pn, self.particle)

class PSOSolver(Solver):
    """Particle Swarm Optimization
    """
    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(PSOSolver, self).__init__(name='pso_vne', reusable=reusable, verbose=verbose, **kwargs)
        """
        0 < p_i < p_c < p_s < 1
        p_i + p_c + p_s = 1
        """
        self.controller = Controller()
        self.counter = Counter()
        # super parameters
        self.p_i = 0.1  # inertia weight
        self.p_c = 0.2  # cognition weight
        self.p_s = 0.7  # social weight
        self.num_particles = 5   # number of num_particles
        self.max_iteration = 30  # max iteration
        # discrete operations
        # sub for positions: Similar to Hamming distance
        # add for positions: Select a location based on probability
        self.sub = lambda pos_x, pos_y:[int(px == py) for px, py in zip(pos_x, pos_y)]
        self.add = lambda p1, pos_1, p2, pos_2, p3, pos_3: [np.random.choice([i,j,k],p=[p1, p2, p3]).tolist() for i, j, k in zip(pos_1, pos_2, pos_3)]

    def solve(self, instance):
        vn, pn  = instance['vn'], instance['pn']
        solution = self.pso(vn, pn)
        if solution['result']:
            self.controller.deploy(vn, pn, solution)
        return solution

    def create_particle_runners(self, vn, pn):
        particle_runners = []
        for i in range(self.num_particles):
            particle_runners.append(ParticleRunner(vn, pn, self, self.particles[i]))
        return particle_runners

    def normalize_values(self, value_list):
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

    def pso(self, vn, pn):
        # initialization
        # construct candidates
        self.candidates_dict = self.controller.construct_candidates_dict(vn, pn)
        if {} in self.candidates_dict.values():
            return Solution(vn)
        # particles
        self.particles = [Particle(i, vn) for i in range(self.num_particles)]
        # global
        self.global_best = {'fitness_value': float('inf'), 'particle': None}
        # initialize particles best experience
        fitness_value_list = []
        for i in range(self.num_particles):
            for v_node_id in range(vn.num_nodes):
                node_position = self.generate_node_position(self.particles[i], v_node_id)
                if node_position is None:
                    break
                else:
                    self.particles[i].position[v_node_id] = node_position
            self.mapping(vn, pn, self.particles[i])
            self.particles[i].best_position = copy.deepcopy(self.particles[i].position)
            self.particles[i].best_solution = copy.deepcopy(self.particles[i].solution)
        fitness_value_list = [self.particles[i].fitness_value for i in range(self.num_particles)]
        if set(fitness_value_list) == (float('inf')):
            return Solution(vn)
        # initialize global best experience
        for i in range(self.num_particles):
            if self.global_best['fitness_value'] >= fitness_value_list[i]:
                self.global_best['fitness_value'] = fitness_value_list[i]
                self.global_best['particle'] = self.particles[i]
        # start iterating
        for id in range(self.max_iteration):
            particle_runners = self.create_particle_runners(vn, pn)
            for particle_runner in particle_runners:
                particle_runner.start()
            for particle_runner in particle_runners:
                particle_runner.join()

            fitness_value_list = [self.particles[i].fitness_value for i in range(self.num_particles)]

            for i in range(self.num_particles):
                if self.particles[i].best_fitness_value > self.particles[i].fitness_value:
                    self.particles[i].best_position = copy.deepcopy(self.particles[i].position)
                    self.particles[i].best_solution = copy.deepcopy(self.particles[i].solution)
                if self.global_best['fitness_value'] > self.particles[i].best_fitness_value:
                    self.global_best['fitness_value'] = self.particles[i].best_fitness_value
                    self.global_best['particle'] = self.particles[i]

        if self.global_best['fitness_value'] == float('inf'):
            return Solution(vn)
        return self.global_best['particle'].best_solution

    def generate_node_position(self, particle, vn_node_id):
        if self.reusable:
            candicate_nodes = self.candidates_dict[vn_node_id]
        else:
            candicate_nodes = list(set(self.candidates_dict[vn_node_id]).difference(set(particle.position)))
        return random.choice(candicate_nodes) if len(candicate_nodes) != 0 else None

    def mapping(self, vn, pn, particle):
        pn = copy.deepcopy(pn)
        if -1 in particle.position:
            return
        # node mapping
        node_mapping_result = self.controller.node_mapping(vn, pn, list(vn.nodes), particle.position, particle.solution, 
                                                            reusable=False, inplace=True, matching_mathod='l2s2')
        if not node_mapping_result:
            return
        # link mapping
        link_mapping_result = self.controller.link_mapping(vn, pn, particle.solution, sorted_v_edges=None,
                                                            shortest_method='bfs_shortest', k=1, inplace=True)
        if not link_mapping_result:
            return 
        # Success
        particle.solution['result'] = True
        Counter.count_solution(vn, particle.solution)
        return