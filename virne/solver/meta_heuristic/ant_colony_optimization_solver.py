# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import threading
import numpy as np
from threading import Thread

from virne.core.environment import SolutionStepEnvironment
from .base_meta_heuristic_solver import Individual, BaseMetaHeuristicSolver
from virne.network import VirtualNetwork, PhysicalNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.solver.base_solver import Solver, SolverRegistry

"""
[CEC, 2017]
Link mapping-oriented ant colony system for virtual network embedding

population size: 20
number of iteration: 10
threshold p0: 0.9
"""


class Ant(Individual):
    
    def __init__(self, id, v_net, p_net):
        super(Ant, self).__init__(id, v_net, p_net)


@SolverRegistry.register(solver_name='aco_meta', solver_type='meta_heuristic')
class AntColonyOptimizationSolver(BaseMetaHeuristicSolver):
    """
    Ant Colony Optimization (ACO) for VNE

    References:
        - Ilhem Fajjari et al. "VNE-AC: Virtual Network Embedding Algorithm Based on Ant Colony Metaheuristic". In ICC, 2011.
        - Hong-Kun Zheng et al. "Link mapping-oriented ant colony system for virtual network embedding". In CEC, 2017.
    
    Attributes:
        num_ants: number of ants
        max_iteration: max iteration
        hop_range: hop-range within local search area
        alpha: control the influence of the amount of pheromone when making a choice in _pick_path()
        beta: control the influence of the distance to the next node in _pick_path()
        coeff_pheromone_evaporation: pheromone evaporation coefficient
        coeff_pheromone_enhancement: enhance pheromone coefficient
        enhence_pheromone: enhance pheromone method, ['best', 'all', 'both']
        node_ranking_method: node ranking method, ['rw', 'dp']
    """
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(AntColonyOptimizationSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        # super parameters
        self.num_ants = 8
        self.max_iteration = 12
        self.hop_range = 2  # hop-range within local search area
        self.alpha = 1.0    # control the influence of the amount of pheromone when making a choice in _pick_path()
        self.beta = 2.0     # control the influence of the distance to the next node in _pick_path()
        self.coeff_pheromone_evaporation = 0.5   # pheromone evaporation coefficient
        self.coeff_pheromone_enhancement = 1.  # enhance pheromone coefficient
        self.enhence_pheromone = 'best'
        # self.node_ranking_method = 'rw'

    def meta_run(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        self.heuristic_info = self.calc_heuristic_info(p_net)
        self.pheromone_trail = self.init_pheromone_trail(self.candidates_dict, value=1.)

        for iter_id in range(self.max_iteration):
            self.ants = [Ant(id, v_net, p_net) for i in range(self.num_ants)]
            ant_runners = []
            for ant in self.ants:
                ant_runners.append(threading.Thread(target=self.evolve, args=(ant, )))
            for ant_runner in ant_runners:
                ant_runner.start()
            for ant_runner in ant_runners:
                ant_runner.join()

            self.update_best_individual(self.ants)
            self.update_pheromone_trail()
        return self.best_individual.best_solution

    def init_pheromone_trail(self, candidates_dict, value=0.):
        pheromone_trail = {}
        for v_node_id in candidates_dict:
            pheromone_trail[v_node_id] = {}
            for p_node_id in candidates_dict[v_node_id]:
                pheromone_trail[v_node_id][p_node_id] = value
        return pheromone_trail

    def calc_heuristic_info(self, p_net):
        node_resource = np.array(p_net.get_node_attrs_data(p_net.get_node_attrs('resource'))).sum(0).tolist()
        bw_sum_resource = np.array(p_net.get_aggregation_attrs_data(p_net.get_link_attrs('resource'), aggr='sum', normalized=False)).sum(0).tolist()
        max_resource = -float('inf')
        min_resource = float('inf')
        heuristic_info = {}
        for v_node_id, candidates in self.candidates_dict.items():
            heuristic_info[v_node_id] = {}
            for p_node_id in candidates:
                heuristic_info[v_node_id][p_node_id] = node_resource[p_node_id] + bw_sum_resource[p_node_id]
                if heuristic_info[v_node_id][p_node_id] > max_resource:
                    max_resource = heuristic_info[v_node_id][p_node_id]
                if heuristic_info[v_node_id][p_node_id] < min_resource:
                    min_resource = heuristic_info[v_node_id][p_node_id]
        # normalization
        for v_node_id in heuristic_info:
            for p_node_id in heuristic_info[v_node_id]:
                heuristic_info[v_node_id][p_node_id] = (heuristic_info[v_node_id][p_node_id] - min_resource) / (max_resource - min_resource)
        return heuristic_info

    def get_v_net_node_order(self, v_net):
        """
        
        TO-DO: Select the virtual node with the highest number of hanging links

        hanging links: virtual links missing one of their outermost virtual nodes, which had already been mapped.
        solution_component = selected virtual node and all its attached hanging links
        """
        ranked_v_nodes = list(range(v_net.num_nodes))
        return ranked_v_nodes
    
    def update_pheromone_trail(self):
        temp_pheromone_trail = self.init_pheromone_trail(self.candidates_dict, value=0.0)
        if self.enhence_pheromone in ['all', 'both']:
            for ant in self.ants:
                if not ant.solution['result']:
                    continue
                for v_node_id in ant.v_net.ranked_nodes:
                    p_node_id = ant.solution['node_slots'][v_node_id]
                    temp_pheromone_trail[v_node_id][p_node_id] += self / ant.fitness
        elif self.enhence_pheromone in ['best', 'both']:
            if self.best_individual.best_solution['result']:
                for v_node_id in self.best_individual.v_net.ranked_nodes:
                    p_node_id = self.best_individual.best_solution['node_slots'][v_node_id]
                    temp_pheromone_trail[v_node_id][p_node_id] += self.coeff_pheromone_enhancement / self.best_individual.best_fitness
        else:
            raise NotImplementedError("enhence_pheromone should be in ['all', 'best', 'both']")

        for v_node_id in self.candidates_dict:
            for p_node_id in self.candidates_dict[v_node_id]:
                self.pheromone_trail[v_node_id][p_node_id] = self.pheromone_trail[v_node_id][p_node_id] * self.coeff_pheromone_evaporation \
                                             + temp_pheromone_trail[v_node_id][p_node_id]

    def select_p_candidate(self, v_node_id, selected_p_nodes):
        p_candidates = self.get_p_condicates(v_node_id, selected_p_nodes)
        if not p_candidates:
            return -1
        else:
            select_probs = {}
            for p_node_id in p_candidates:
                weighted_pheromone = pow(self.pheromone_trail[v_node_id][p_node_id], self.alpha)
                weighted_heuristic = pow(self.heuristic_info[v_node_id][p_node_id], self.beta)
                select_probs[p_node_id] = weighted_pheromone * weighted_heuristic
            probs = np.array(list(select_probs.values()))
            norm_probs = probs / (probs.sum())
            p_node_id = np.random.choice(np.array(list(select_probs.keys())), p=norm_probs)
            return p_node_id

    def evolve(self, ant):
        for v_node_id in ant.v_net.ranked_nodes:
            p_node_id = self.select_p_candidate(v_node_id, ant.selected_p_nodes)
            if p_node_id != -1:
                result, info = self.controller.place_and_route(ant.v_net, ant.p_net, v_node_id, p_node_id, ant.solution, shortest_method='bfs_shortest', k=1)
            else:
                break
            if not result:
                break
        # success
        if len(ant.v_net.ranked_nodes) == len(ant.solution['node_slots']) and result:
            ant.solution['result'] = True
            self.counter.count_solution(ant.v_net, ant.solution)
            ant.update_best_solution()


    # def get_node_placement_order(self, v_net, ant):
    #     """Select the virtual node with the highest number of hanging links

    #     hanging links: virtual links missing one of their outermost virtual nodes, which had already been mapped.
    #     solution_component = selected virtual node and all its attached hanging links
    #     """
    #     num_hanging_link_dict = {}
    #     placed_v_net_nodes = ant.solution['node_slots'].keys()

    #     for v_node_id in range(v_net.num_nodes):
    #         if v_node_id in placed_v_net_nodes:
    #             continue
    #         v_node_id_neighbors = list(v_net.adj[v_node_id])
    #         num_hanging_links = sum([1 if neighbor not in placed_v_net_nodes else 0 for neighbor in v_node_id_neighbors])
    #         num_hanging_link_dict[v_node_id] = num_hanging_links

    #     assert num_hanging_link_dict

    #     v_node_id = max(num_hanging_link_dict, key=num_hanging_link_dict.get)
    #     return v_node_id
