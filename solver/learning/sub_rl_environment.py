import gym
import copy
import numpy as np
import networkx as nx

from gym import spaces

from .obs_handler import ObservationHandler
from base.controller import Controller
from base.recorder import Solution, Counter
from data import PhysicalNetwork, VNSimulator


class SubRLEnv(gym.Env):

    def __init__(self, pn, vn, rejection_action=False, reusable=False):
        super().__init__()
        self.rejection_action = rejection_action
        self.reusable = reusable
        self.pn = pn
        self.curr_vn = vn
        self.pn_backup = copy.deepcopy(pn)
        self.curr_solution = Solution(vn)
        self.curr_vnf_id = 0
        self.counter = Counter()
        self.controller = Controller()
        self.obs_handler = ObservationHandler()
        self.action_space = spaces.Discrete(self.pn.num_nodes + 1 if rejection_action else self.pn.num_nodes)
        self.node_extrema_data, self.node_attrs_benchmark = self.obs_handler.get_node_attrs_obs(self.pn, node_attr_types=['extrema'], normalization=True)
        self.edge_extrema_data, self.edge_attrs_benchmark = self.obs_handler.get_edge_attrs_obs(self.pn, edge_attr_types=['extrema'], normalization=True)
        self.edge_aggr_extrema_data, self.edge_aggr_attrs_benchmark = self.obs_handler.get_edge_aggr_attrs_obs(self.pn, edge_attr_types=['extrema'], normalization=True)
        self.curr_vn_reward = 0
        # Graph theory features
        # degree
        pn_node_degrees = np.array([list(nx.degree_centrality(self.pn).values())]).T
        self.pn_node_degrees = (pn_node_degrees - pn_node_degrees.min()) / (pn_node_degrees.max() - pn_node_degrees.min())
        # closeness
        pn_node_closenesses = np.array([list(nx.closeness_centrality(self.pn).values())]).T
        self.pn_node_closenesses = (pn_node_closenesses - pn_node_closenesses.min()) / (pn_node_closenesses.max() - pn_node_closenesses.min())
        # eigenvector
        pn_node_eigenvectors = np.array([list(nx.eigenvector_centrality(self.pn).values())]).T
        self.pn_node_eigenvectors = (pn_node_eigenvectors - pn_node_eigenvectors.min()) / (pn_node_eigenvectors.max() - pn_node_eigenvectors.min())
        # betweenness
        pn_node_betweennesses = np.array([list(nx.betweenness_centrality(self.pn).values())]).T
        self.pn_node_betweennesses = (pn_node_betweennesses - pn_node_betweennesses.min()) / (pn_node_betweennesses.max() - pn_node_betweennesses.min())


    def reset(self):
        self.curr_solution.reset()
        self.pn = copy.deepcopy(self.pn_backup)
        self.curr_vnf_id = 0
        return super().reset()

    def step(self, action):
        pass

    def get_observation(self):
        pass

    def compute_reward(self):
        pass

    def get_info(self, info={}):
        info = copy.deepcopy(info)
        return info

    def render(self):
        return

    def generate_action_mask(self):
        candidate_nodes = self.controller.find_candidate_nodes(self.pn, self.curr_vn, self.curr_vnf_id, filter=self.selected_pn_nodes)
        mask = np.zeros(self.pn.num_nodes, dtype=bool)
        mask[candidate_nodes] = 1
        if mask.sum() == 0: mask[0] = True
        return mask

    def get_curr_place_progress(self):
        return self.curr_vnf_id / self.curr_vn.num_nodes

    def get_node_load_balance(self, p_node_id):
        n_attrs = self.pn.get_node_attrs(['resource'])
        if len(n_attrs) > 1:
            n_resources = np.array([self.pn.nodes[p_node_id][n_attr.name] for n_attr in n_attrs])
            load_balance = np.std(n_resources)
        else:
            n_attr = self.pn.get_node_attrs(['extrema'])[0]
            load_balance = self.pn.nodes[p_node_id][n_attr.originator] / self.pn.nodes[p_node_id][n_attr.name]
        return load_balance

    @property
    def selected_pn_nodes(self):
        return list(self.curr_solution['node_slots'].values())

    @property
    def placed_vn_nodes(self):
        return list(self.curr_solution['node_slots'].keys())



class SolutionStepSubRLEnv(SubRLEnv):
    
    def __init__(self, pn, vn, rejection_action=False, reusable=False, **kwargs):
        super(SolutionStepSubRLEnv, self).__init__(pn, vn, rejection_action, reusable, **kwargs)

    def step(self, solution):
        # Success
        if solution['result']:
            self.curr_solution = solution
            self.curr_solution['description'] = 'Success'
        # Failure
        else:
            solution = Solution(self.curr_vn)
        return self.get_observation(), self.compute_reward(), True, self.get_info(solution.__dict__)

    def get_info(self, record={}):
        info = copy.deepcopy(record)
        return info

    def get_observation(self):
        return {'vn': self.curr_vn, 'pn': self.pn}

    def compute_reward(self):
        return 0

    def generate_action_mask(self):
        return np.arange(self.pn.num_nodes)


class JointPRStepSubRLEnv(SubRLEnv):
    
    def __init__(self, pn, vn, rejection_action=False, reusable=False, **kwargs):
        super(JointPRStepSubRLEnv, self).__init__(pn, vn, rejection_action, reusable, **kwargs)

    def step(self, action):
        """
        Joint Place and Route with action pn node.

        All possible case
            Uncompleted Success: (Node place and Link route successfully)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (Node place failed or Link route failed)
        """
        p_node_id = int(action)
        done = True
        # Case: Reject Actively
        if self.rejection_action and p_node_id == self.pn.num_nodes:
            self.curr_solution['early_rejection'] = True
            solution_info = self.curr_solution.to_dict()
        # Case: Place in one same node
        elif not self.reusable and p_node_id in self.selected_pn_nodes:
            self.curr_solution['place_result'] = False
            solution_info = self.curr_solution.to_dict()
        # Case: Try to Place and Route
        else:
            assert p_node_id in list(self.pn.nodes)
            place_and_route_result = self.controller.place_and_route(self.curr_vn, self.pn, self.curr_vnf_id, p_node_id,
                                                                        self.curr_solution, shortest_method='bfs_shortest', k=50)
            # Step Failure
            if not place_and_route_result:
                solution_info = self.curr_solution.to_dict()
            else:
                self.curr_vnf_id += 1
                # VN Success ?
                if self.curr_vnf_id == self.curr_vn.num_nodes:
                    self.curr_solution['result'] = True
                    solution_info = self.counter.count_solution(self.curr_vn, self.curr_solution)
                # Step Success
                else:
                    done = False
                    solution_info = self.counter.count_partial_solution(self.curr_vn, self.curr_solution)
                    
        if done:
            self.curr_vnf_id = 0
        return self.get_observation(), self.compute_reward(self.curr_solution), done, self.get_info(solution_info)


class PlaceStepSubRLEnv(SubRLEnv):

    def __init__(self, pn, vn, rejection_action=False, reusable=False, **kwargs):
        super(PlaceStepSubRLEnv, self).__init__(pn, vn, rejection_action, reusable, **kwargs)

    def step(self, action):
        """
        Two stage: Node Mapping and Link Mapping

        All possible case
            Early Rejection: (rejection_action)
            Uncompleted Success: (Node place)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (not Node place, not Link mapping)
        """
        p_node_id = int(action)
        done = True
        # Case: Reject Actively
        if self.rejection_action and p_node_id == self.pn.num_nodes:
            self.curr_solution['early_rejection'] = True
            solution_info = self.curr_solution.to_dict()
        # Case: Place in one same node
        elif not self.reusable and p_node_id in self.selected_pn_nodes:
            self.curr_solution['place_result'] = False
            solution_info = self.curr_solution.to_dict()
        # Case: Try to Place
        else:
            assert p_node_id in list(self.pn.nodes)
            # Stage 1: Node Mapping
            node_place_result = self.controller.place(self.curr_vn, self.pn, self.curr_vnf_id, p_node_id, self.curr_solution)
            self.curr_vnf_id += 1
            # Case 1: Node Place Success / Uncompleted
            if node_place_result and self.curr_vnf_id < self.curr_vn.num_nodes:
                done = False
                solution_info = self.curr_solution.to_dict()
                return self.get_observation(), self.compute_reward(self.curr_solution), False, self.get_info(self.curr_solution.to_dict())
            # Case 2: Node Place Failure
            if not node_place_result:
                self.curr_solution['place_result'] = False
                solution_info = self.curr_solution.to_dict()
            # Stage 2: Link Mapping
            # Case 3: Try Link Mapping
            if node_place_result and self.curr_vnf_id == self.curr_vn.num_nodes:
                link_mapping_result = self.controller.link_mapping(self.curr_vn, self.pn, solution=self.curr_solution, sorted_v_edges=list(self.curr_vn.edges), 
                                                                    shortest_method='bfs_shortest', k=50, inplace=True)
                # Link Mapping Failure
                if not link_mapping_result:
                    self.curr_solution['route_result'] = False
                    solution_info = self.curr_solution.to_dict()
                # Success
                else:
                    self.curr_solution['result'] = True
                    solution_info = self.counter.count_solution(self.curr_vn, self.curr_solution)
        if done:
            self.curr_vnf_id = 0
        return self.get_observation(), self.compute_reward(solution_info), done, self.get_info(solution_info)
