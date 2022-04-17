import copy
import numpy as np
from threading import Thread

from base import Controller, Solution, Counter
from ..solver import Solver
from ..rank.node_rank import OrderNodeRank, GRCNodeRank, FFDNodeRank, NRMNodeRank, RWNodeRank, RandomNodeRank
from ..rank.edge_rank import OrderEdgeRank, FFDEdgeRank


class Ant(Thread):
    
    def __init__(self, id, vn, pn, candidates_dict, heuristic_info, pheromone_trail, sorted_v_nodes, alpha, beta):
        Thread.__init__(self)
        self.id = id
        self.vn = vn
        self.pn_backup = copy.deepcopy(pn)
        self.candidates_dict = candidates_dict
        self.heuristic_info = heuristic_info
        self.pheromone_trail = pheromone_trail
        self.sorted_v_nodes = sorted_v_nodes

        self.alpha = alpha
        self.beta = beta

        self.controller = Controller()
        self.counter = Counter()

        self.reset()

    @property
    def selected_pn_nodes(self, ):
        return list(self.solution['node_slots'].values())

    @property
    def total_cost(self, ):
        return self.calc_total_cost(self.solution)

    def reset(self):
        self.pn = copy.deepcopy(self.pn_backup)
        self.solution = Solution(self.vn)

    def calc_total_cost(self, solution):
        # minimize
        if solution['result']:
            return solution['vn_cost']
        return float('inf')

    def select_pn_node(self, v_node_id):
        candidate_nodes = list(set(self.candidates_dict[v_node_id]).difference(set(self.selected_pn_nodes)))
        if not candidate_nodes:
            return None
        else:
            select_probs = {}
            for p_node_id in candidate_nodes:
                weighted_pheromone = pow(self.pheromone_trail[v_node_id][p_node_id], self.alpha)
                weighted_heuristic = pow(self.heuristic_info[v_node_id][p_node_id], self.beta)
                select_probs[p_node_id] = weighted_pheromone * weighted_heuristic
            probs = np.array(list(select_probs.values()))
            norm_probs = probs / (probs.sum())
            p_node_id = np.random.choice(np.array(list(select_probs.keys())), p=norm_probs)
            return p_node_id

    def run(self):
        self.reset()
        for v_node_id in self.sorted_v_nodes:
            p_node_id = self.select_pn_node(v_node_id)
            if p_node_id is not None:
                result = self.controller.place_and_route(self.vn, self.pn, v_node_id, p_node_id, self.solution, shortest_method='bfs_shortest', k=1)
            else:
                break
        if len(self.sorted_v_nodes) == len(self.solution['node_slots']):
            self.solution['result'] = True
            self.counter.count_solution(self.vn, self.solution)


class ACOSolver(Solver):
    """Ant Colony Optimization
    """
    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(ACOSolver, self).__init__('aco_vne', reusable, verbose, **kwargs)
        self.controller = Controller()
        self.counter = Counter()
        # super parameters
        self.num_ants = 10
        self.max_iteration = 20
        self.hop_range = 2  # hop-range within local search area
        self.alpha = 1.0    # control the influence of the amount of pheromone when making a choice in _pick_path()
        self.beta = 2.0     # control the influence of the distance to the next node in _pick_path()
        self.coeff_pheromone_evaporation = 0.5  # pheromone evaporation coefficient
        self.coeff_pheromone_enhancement = 100.  # enhance pheromone coefficient
        self.enhence_pheromone = 'best'  # ['all', 'best', 'both']

    def solve(self, instance):
        vn, pn  = instance['vn'], instance['pn']
        solution = self.aco(vn, pn)
        if solution['result']:
            self.controller.deploy(vn, pn, solution)
        return solution

    def aco(self, vn, pn):
        # init
        self.candidates_dict = self.construct_candidates_dict(vn, pn)
        if {} in self.candidates_dict.values():
            return Solution(vn)

        self.heuristic_info = self.calc_heuristic_info(pn)
        self.pheromone_trail = self.init_pheromone_trail(self.candidates_dict, value=1.)
        self.sorted_v_nodes = self.get_vn_node_order(vn)

        self.global_best = {'solution': Solution(vn), 'total_cost': float('inf')}

        for iter_id in range(self.max_iteration):
            self.ants = self.create_ants(vn, pn)
            for ant in self.ants:
                ant.start()
            for ant in self.ants:
                ant.join()

            for ant in self.ants:
                if ant.total_cost < self.global_best['total_cost']:
                    self.global_best['solution'] = copy.deepcopy(ant.solution)
                    self.global_best['total_cost'] = ant.total_cost
            self.update_pheromone_trail()
        return self.global_best['solution']

    def create_ants(self, vn, pn):
        ants = [Ant(id, vn, pn,
                self.candidates_dict, 
                self.heuristic_info, 
                self.pheromone_trail, 
                self.sorted_v_nodes, 
                self.alpha, self.beta) 
                for i in range(self.num_ants)
        ]
        return ants

    def init_pheromone_trail(self, candidates_dict, value=0.):
        pheromone_trail = {}
        for v_node_id in candidates_dict:
            pheromone_trail[v_node_id] = {}
            for p_node_id in candidates_dict[v_node_id]:
                pheromone_trail[v_node_id][p_node_id] = value
        return pheromone_trail

    def construct_candidates_dict(self, vn, pn):
        candidates_dict = self.controller.construct_candidates_dict(vn, pn)
        # for v_node_id in range(vn.num_nodes):
        #     hop_range_neighbors = nx.single_source_shortest_path_length(pn, p_node_id, cutoff=self.hop_range)
        #     for candidate in candidates_dict[v_node_id]:
        #         if candidate not in hop_range_neighbors:
        #             candidates_dict[v_node_id].remove(candidate)
        return candidates_dict

    def calc_heuristic_info(self, pn):
        node_resource = np.array(pn.get_node_attrs_data(pn.get_node_attrs('resource'))).sum(0).tolist()
        bw_sum_resource = np.array(pn.get_aggregation_attrs_data(pn.get_edge_attrs('resource'), aggr='sum', normalized=False)).sum(0).tolist()
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
        # for v_node_id in heuristic_info:
        #     for p_node_id in heuristic_info[v_node_id]:
        #         heuristic_info[v_node_id][p_node_id] = (heuristic_info[v_node_id][p_node_id] - min_resource) / (max_resource - min_resource)
        return heuristic_info

    def get_vn_node_order(self, vn):
        """
        
        TO-DO: Select the virtual node with the highest number of hanging links

        hanging links: virtual links missing one of their outermost virtual nodes, which had already been mapped.
        solution_component = selected virtual node and all its attached hanging links
        """
        sorted_v_nodes = list(range(vn.num_nodes))
        return sorted_v_nodes
    
    def update_pheromone_trail(self):
        temp_pheromone_trail = self.init_pheromone_trail(self.candidates_dict, value=0.0)
        if self.enhence_pheromone in ['all', 'both']:
            for ant in self.ants:
                if not ant.solution['result']:
                    continue
                for v_node_id in self.sorted_v_nodes:
                    p_node_id = ant.solution['node_slots'][v_node_id]
                    temp_pheromone_trail[v_node_id][p_node_id] += self / ant.total_cost
        elif self.enhence_pheromone in ['best', 'both']:
            if self.global_best['solution']['result']:
                for v_node_id in self.sorted_v_nodes:
                    p_node_id = self.global_best['solution']['node_slots'][v_node_id]
                    temp_pheromone_trail[v_node_id][p_node_id] += self.coeff_pheromone_enhancement / self.global_best['total_cost']
        else:
            raise NotImplementedError("enhence_pheromone should be in ['all', 'best', 'both']")

        for v_node_id in self.candidates_dict:
            for p_node_id in self.candidates_dict[v_node_id]:
                self.pheromone_trail[v_node_id][p_node_id] = self.pheromone_trail[v_node_id][p_node_id] * self.coeff_pheromone_evaporation \
                                             + temp_pheromone_trail[v_node_id][p_node_id]


    # def get_node_placement_order(self, vn, ant):
    #     """Select the virtual node with the highest number of hanging links

    #     hanging links: virtual links missing one of their outermost virtual nodes, which had already been mapped.
    #     solution_component = selected virtual node and all its attached hanging links
    #     """
    #     num_hanging_link_dict = {}
    #     placed_vn_nodes = ant.solution['node_slots'].keys()

    #     for v_node_id in range(vn.num_nodes):
    #         if v_node_id in placed_vn_nodes:
    #             continue
    #         v_node_id_neighbors = list(vn.adj[v_node_id])
    #         num_hanging_links = sum([1 if neighbor not in placed_vn_nodes else 0 for neighbor in v_node_id_neighbors])
    #         num_hanging_link_dict[v_node_id] = num_hanging_links

    #     assert num_hanging_link_dict

    #     vn_node_id = max(num_hanging_link_dict, key=num_hanging_link_dict.get)
    #     return vn_node_id
