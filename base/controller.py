import copy
import pprint
import numpy as np
import networkx as nx
from itertools import islice
from collections import deque
from ortools.linear_solver import pywraplp
from ortools.util.python import sorted_interval_list
Domain = sorted_interval_list.Domain

from .solution import Solution
from .utils import flatten_recurrent_dict
from data.attribute import create_attrs_from_setting


'''
Methods List

### ---------- 1. Attribute-Level --------- ###

1.1 CHECK
We provide thress methods to check (node, link and path)-level constraints,
the final result and satisfiability information will be return.
    check_node_constraints
    check_link_constraints
    check_path_constraints

1.2 UPDATE
We implement thress methods to update (node, link and path)-level modifiable attributes, such as resources,
specifically, the attributes will be directly updated without any verification.
    update_node_resources
    update_link_resources
    update_path_resources

### ---------- 2. Place & Route-Level --------- ###

place
route
place_and_route

unsafely_place
unsafely_route
unsafely_place_and_route

undo_place
undo_route
undo_place_and_route

### ---------- 3. Mapping-Level --------- ###

node_mapping
link_mapping

### ---------- 4. Deploy-Level --------- ###

deploy
release
bfs_deploy
undo_deploy: release and reset the solution
'''


class Controller:
    
    latency_constraint_name = 'latency'

    def __init__(self, counter, node_attrs_setting=[], link_attrs_setting=[], **kwargs):
        self.counter = counter
        self.all_node_attrs = list(create_attrs_from_setting(node_attrs_setting).values())
        self.all_link_attrs = list(create_attrs_from_setting(link_attrs_setting).values())
        self.node_resource_attrs = [n_attr for n_attr in self.all_node_attrs if n_attr.type == 'resource']
        self.link_resource_attrs = [l_attr for l_attr in self.all_link_attrs if l_attr.type == 'resource']
        self.link_latency_attrs = [l_attr for l_attr in self.all_link_attrs if l_attr.type == 'latency']
        self.reusable = kwargs.get('reusable', False)
        # ranking strategy
        self.node_ranking_method = kwargs.get('node_ranking_method', 'order')
        self.link_ranking_method = kwargs.get('link_ranking_method', 'order')
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'k_shortest')

    def check_attributes(self, v, p, attrs_list):
        final_result = True       # check result
        satisfiability_info = {}  # 
        for attr in attrs_list:
            result, value = attr.check(v, p)
            if not result:
                final_result = False
            # record the resource violations
            if attr.type == 'resource':
                satisfiability_info[attr.name] = value
            else:
                satisfiability_info[attr.name] = 0.
        return final_result, satisfiability_info

    def check_graph_constraints(self, v_net, p_net):
        final_result, graph_satisfiability_info = self.check_attributes(v_net, p_net, self.all_node_attrs)
        return final_result, graph_satisfiability_info

    def check_node_constraints(self, v_net, p_net, v_node_id, p_node_id):
        assert p_node_id in list(p_net.nodes)
        v_node, p_node = v_net.nodes[v_node_id], p_net.nodes[p_node_id]
        final_result, node_satisfiability_info = self.check_attributes(v_node, p_node, self.all_node_attrs)
        return final_result, node_satisfiability_info

    def check_link_constraints(self, v_net, p_net, v_link_pair, p_link_pair):
        v_link, p_link = v_net.links[v_link_pair], p_net.links[p_link_pair]
        final_result, link_satisfiability_info = self.check_attributes(v_link, p_link, self.all_link_attrs)
        return final_result, link_satisfiability_info

    def check_path_constraints(self, v_net, p_net, v_link, p_path):
        p_links = self.path_to_links(p_path)
        final_result = True
        path_satisfiability_info = dict()
        for p_link in p_links:
            result, info = self.check_link_constraints(v_net, p_net, v_link, p_link)
            if not result:
                final_result = False
            path_satisfiability_info[p_link] = info
        # TODO: Add path constraint
        return final_result, path_satisfiability_info

    def update_resource(self, net, element_owner, element_id, attr_name, value, operator='-', safe=True):
        assert operator in ['+', '-', 'add', 'sub']
        assert element_owner in ['node', 'link']
        if operator in ['+', 'add']:
            if element_owner == 'node':
                net.nodes[element_id][attr_name] += value
            elif element_owner == 'link':
                net.links[element_id][attr_name] += value
        elif operator in ['-', 'sub']:
            if element_owner == 'node':
                if safe: assert net.nodes[element_id][attr_name] >= value, f"Node {element_id} and Attribute {attr_name}: {net.nodes[element_id][attr_name]} - {value}"
                net.nodes[element_id][attr_name] -= value
            elif element_owner == 'link':
                if safe: assert net.links[element_id][attr_name] >= value
                net.links[element_id][attr_name] -= value
        else:
            raise NotImplementedError

    def update_node_resources(self, p_net, p_node_id, used_node_resources, operator='-', safe=True):
        for n_attr_name, value in used_node_resources.items():
            self.update_resource(p_net, 'node', p_node_id, n_attr_name, value, operator=operator, safe=safe)

    def update_link_resources(self, p_net, p_link, used_link_resources, operator='-', safe=True):
        for e_attr_name, value in used_link_resources.items():
            self.update_resource(p_net, 'link', p_link, e_attr_name, value, operator=operator, safe=safe)

    def update_path_resources(self, v_net, p_net, v_link, p_path, operator='-', safe=True):
        for l_attr in self.link_resource_attrs:
            l_attr.update_path(v_net.links[v_link], p_net, p_path, operator, safe=safe)

    def place(self, v_net, p_net, v_node_id, p_node_id, solution):
        r"""Attempt to place the VNF `v_node_id` in PN node `pid`."""
        check_result, check_info = self.check_node_constraints(v_net, p_net, v_node_id, p_node_id)
        if not check_result:
            return False, check_info
        used_node_resources = {n_attr.name: v_net.nodes[v_node_id][n_attr.name] for n_attr in self.node_resource_attrs}
        self.update_node_resources(p_net, p_node_id, used_node_resources, operator='-')
        solution['node_slots'][v_node_id] = p_node_id
        solution['node_slots_info'][(v_node_id, p_node_id)] = used_node_resources
        return True, check_info

    def route(self, v_net, p_net, v_link, pl_pair, solution=None, shortest_method='all_shortest', k=1, rank_path_func=None):
        r"""Return True if route successfully the virtual link (vid_a, v_node_id_b) in the physical network path (pid_a, p_node_id_b); Otherwise False.
        
            shortest_method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        # currently, only first_shortest, k_shortest and all_shortest support unsafe routing mode

        # place the prev VNF and curr VNF on the identical physical node
        check_info = {l_attr.name: 0. for l_attr in v_net.get_link_attrs()}
        if pl_pair[0] == pl_pair[1]:
            if self.reusable:
                return True, check_info
            else:
                raise NotImplementedError

        shortest_paths = self.find_shortest_paths(v_net, p_net, v_link, pl_pair, method=shortest_method, k=k)
        shortest_paths = rank_path_func(v_net, p_net, v_link, pl_pair, shortest_paths) if rank_path_func is not None else shortest_paths
        check_info_list = []
        for p_path in shortest_paths:
            check_result, check_info = self.check_path_constraints(v_net, p_net, v_link, p_path)
            if check_result:
                p_links = self.path_to_links(p_path)
                solution['link_paths'][v_link] = p_links
                for p_link in p_links:
                    used_link_resources = {l_attr.name: v_net.links[v_link][l_attr.name] for l_attr in self.link_resource_attrs}
                    self.update_link_resources(p_net, p_link, used_link_resources, operator='-', safe=True)
                    solution['link_paths_info'][(v_link, p_link)] = used_link_resources
                return True, check_info
        return False, check_info

    def place_and_route(self, v_net, p_net, v_node_id, p_node_id, solution, shortest_method='bfs_shortest', k=1):
        r"""Attempt to place the VNF `v_node_id` in PN node`pid` 
            and route VLs related to the VNF.

            shortest_method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        # Place
        place_result, place_info = self.place(v_net, p_net, v_node_id, p_node_id, solution)
        if not place_result:
            solution['result'] = False
            solution['place_result'] = False
            return False, place_info
        # Route
        route_info = {l_attr.name: 0. for l_attr in v_net.get_link_attrs()}
        to_route_v_links = []
        v_node_id_neighbors = list(v_net.adj[v_node_id])
        for n_v_node_id in v_node_id_neighbors:
            placed = n_v_node_id in solution['node_slots'].keys() and solution['node_slots'][n_v_node_id] != -1
            routed = (n_v_node_id, v_node_id) in solution['link_paths'].keys() or (v_node_id, n_v_node_id) in solution['link_paths'].keys()
            if placed and not routed:
                to_route_v_links.append((v_node_id, n_v_node_id))
        
        if shortest_method == 'mcf':
            route_result, route_info = self.route_v_links_with_mcf(v_net, p_net, to_route_v_links, solution)
            if not route_result:
                # FAILURE
                solution.update({'route_result': False, 'result': False})
                return False, route_info
        else:
            for v_link in to_route_v_links:
                n_p_node_id = solution['node_slots'][v_link[1]]
                route_result, route_info = self.route(v_net, p_net, v_link, (p_node_id, n_p_node_id), solution, 
                                            shortest_method=shortest_method, k=k)
                if not route_result:
                    solution['result'] = False
                    solution['route_result'] = False
                    return False, route_info

        return True, route_info

    def unsafely_place(self, v_net, p_net, v_node_id, p_node_id, solution):
        r"""Attempt to place the VNF `v_node_id` in PN node `pid`."""
        check_result, check_info = self.check_node_constraints(v_net, p_net, v_node_id, p_node_id)
        used_node_resources = {n_attr.name: v_net.nodes[v_node_id][n_attr.name] for n_attr in self.node_resource_attrs}
        self.update_node_resources(p_net, p_node_id, used_node_resources, operator='-', safe=False)
        solution['node_slots'][v_node_id] = p_node_id
        solution['node_slots_info'][(v_node_id, p_node_id)] = used_node_resources
        if check_result:
            violation_value = 0.
        else:
            all_satisfiability_value_list = flatten_recurrent_dict(check_info)
            all_violation_value_list = [v  if v < 0. else 0 for v in all_satisfiability_value_list]
            violation_value = - sum(all_violation_value_list)
        solution['violation'] += violation_value
        solution['total_place_violation'] += violation_value
        solution['current_violation'] = violation_value
        return True, check_info, violation_value

    def unsafely_route(self, v_net, p_net, v_link, pl_pair, solution, shortest_method='k_shortest', k=10, pruning_ratio=None):
        r"""Return True if route successfully the virtual link (vid_a, v_node_id_b) in the physical network path (pid_a, p_node_id_b); Otherwise False.
        
            shortest_method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        # place the prev VNF and curr VNF on the identical physical node
        check_info = {l_attr.name: 0. for l_attr in v_net.get_link_attrs()}
        if pl_pair[0] == pl_pair[1]:
            if self.reusable:
                return True, check_info
            else:
                raise NotImplementedError
        if pruning_ratio is None:
            pruned_p_net = p_net
        else:
            pruned_p_net = self.create_pruned_network(v_net, p_net, v_link_pair=v_link, ratio=pruning_ratio, div=0)
        shortest_paths = self.find_shortest_paths(v_net, pruned_p_net, v_link, pl_pair, method=shortest_method, k=k)
        if len(shortest_paths) == 0:
            return False, check_info, 0.
        check_result_list = []
        check_info_list = []
        violation_value_list = []
        # check shortest paths
        for p_path in shortest_paths:
            check_result, check_info = self.check_path_constraints(v_net, p_net, v_link, p_path)
            check_info_list.append(check_info)
            check_result_list.append(check_result)
            # exist a feasible path
            if check_result:
                p_links = self.path_to_links(p_path)
                solution['link_paths'][v_link] = p_links
                for p_link in p_links:
                    used_link_resources = {l_attr.name: v_net.links[v_link][l_attr.name] for l_attr in self.link_resource_attrs}
                    self.update_link_resources(p_net, p_link, used_link_resources, operator='-', safe=True)
                    solution['link_paths_info'][(v_link, p_link)] = used_link_resources
                return True, check_info, 0.
        # check_info_dict = {p_path: {p_link: {result}}}
        for check_info in check_info_list:
            all_satisfiability_value_list = list(flatten_recurrent_dict(check_info))
            p_path_violation_value_list = [v  if v < 0. else 0 for v in all_satisfiability_value_list]
            violation_value = -sum(p_path_violation_value_list)
            violation_value_list.append(violation_value)
        # select the best paths with the least violations
        best_p_path_index = violation_value_list.index(min(violation_value_list))
        best_p_path = shortest_paths[best_p_path_index]
        best_violation_value = violation_value_list[best_p_path_index]
        # update best path resources
        p_links = self.path_to_links(best_p_path)
        solution['link_paths'][v_link] = p_links
        for p_link in p_links:
            used_link_resources = {l_attr.name: v_net.links[v_link][l_attr.name] for l_attr in self.link_resource_attrs}
            self.update_link_resources(p_net, p_link, used_link_resources, operator='-', safe=False)
            solution['link_paths_info'][(v_link, p_link)] = used_link_resources
        solution['violation'] += best_violation_value
        solution['total_route_violation'] += best_violation_value
        solution['current_violation'] = best_violation_value
        return True, check_info, best_violation_value

    def unsafely_place_and_route(self, v_net, p_net, v_node_id, p_node_id, solution, shortest_method='bfs_shortest', k=1) -> bool:
        r"""Attempt to place the VN node `v_node_id` in PN node`p_node_id` 
            and route VLs related to the VNF.

            shortest_method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']

        """
        assert shortest_method == 'k_shortest'
        # Place
        place_result, place_info, place_violation_value = self.unsafely_place(v_net, p_net, v_node_id, p_node_id, solution)
        # Route
        route_info = {l_attr.name: 0. for l_attr in v_net.get_link_attrs()}
        to_route_v_links = []
        v_node_id_neighbors = list(v_net.adj[v_node_id])
        route_violation_value_list = []
        for n_v_node_id in v_node_id_neighbors:
            placed = n_v_node_id in solution['node_slots'].keys() and solution['node_slots'][n_v_node_id] != -1
            routed = (n_v_node_id, v_node_id) in solution['link_paths'].keys() or (v_node_id, n_v_node_id) in solution['link_paths'].keys()
            if placed and not routed:
                to_route_v_links.append((v_node_id, n_v_node_id))
        for v_link in to_route_v_links:
            n_p_node_id = solution['node_slots'][v_link[1]]
            route_result, route_info, route_violation_value = self.unsafely_route(v_net, p_net, v_link, (p_node_id, n_p_node_id), solution, 
                                        shortest_method=shortest_method, k=k)
            route_violation_value_list.append(route_violation_value)

        best_violation_value = place_violation_value + sum(route_violation_value_list)
        solution['current_violation'] = best_violation_value
        return True, route_info, best_violation_value

    def undo_place(self, v_node_id, p_net, solution):
        assert v_node_id in solution['node_slots'].keys()
        p_node_id = solution['node_slots'][v_node_id]
        used_node_resources = solution['node_slots_info'][(v_node_id, p_node_id)]
        self.update_node_resources(p_net, p_node_id, used_node_resources, operator='+')
        del solution['node_slots'][v_node_id]
        del solution['node_slots_info'][(v_node_id, p_node_id)]
        return True

    def undo_route(self, v_link, p_net, solution):
        assert v_link in solution['link_paths'].keys()
        p_links = solution['link_paths'][v_link]
        for p_link in p_links:
            used_link_resources = solution['link_paths_info'][(v_link, p_link)]
            self.update_link_resources(p_net, p_link, used_link_resources, operator='+')
            del solution['link_paths_info'][(v_link, p_link)]
        del solution['link_paths'][v_link]
        return True

    def undo_place_and_route(self, v_net, p_net, v_node_id, p_node_id, solution):
        # Undo place
        origin_node_slots = list(solution['node_slots'].keys())
        if v_node_id not in origin_node_slots:
            raise ValueError
        undo_place_result = self.undo_place(v_node_id, p_net, solution)
        # Undo route
        origin_link_paths = list(solution['link_paths'].keys())
        for v_link in origin_link_paths:
            if v_node_id in v_link:
                undo_route_result = self.undo_route(v_link, p_net, solution)
        return True

    def undo_deploy(self, v_net, p_net, solution):
        r"""Release occupied resources when a VN leaves PN, and reset the solution."""
        self.release(v_net, p_net, solution)
        solution.reset()
        return True

    def bfs_deploy(self, v_net, p_net, sorted_v_nodes, p_net_initial_node, max_visit=100, max_depth=10, shortest_method='all_shortest', k=10):
        r"""Deploy the `v_net` in `p_net` starting from `initial_node` using Breadth-First Search solver.

        method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        solution = Solution(v_net)

        max_visit_in_every_depth = int(np.power(max_visit, 1 / max_depth))
        
        curr_depth = 0
        visited = p_net.num_nodes * [False]
        queue = [(p_net_initial_node, curr_depth)]
        visited[p_net_initial_node] = True

        num_placed_nodes = 0
        v_node_id = sorted_v_nodes[num_placed_nodes]
        num_attempt_times = 0
        while queue:
            (curr_pid, depth) = queue.pop(0)
            if depth > max_depth:
                break
            
            place_and_route_reult, place_and_route_info = self.place_and_route(v_net, p_net, v_node_id, curr_pid, solution, shortest_method=shortest_method, k=k)
            if place_and_route_reult:
                num_placed_nodes = num_placed_nodes + 1

                if num_placed_nodes >= len(sorted_v_nodes):
                    solution['result'] = True
                    solution['num_attempt_times'] = num_attempt_times
                    return solution
                v_node_id = sorted_v_nodes[num_placed_nodes]
            else:
                num_attempt_times += 1
                if v_node_id in solution['node_slots']:
                    self.undo_place_and_route(v_net, p_net, v_node_id, curr_pid, solution)

            if depth == max_depth:
                continue

            node_links = p_net.links(curr_pid, data=False)
            node_links = node_links if len(node_links) <= max_visit else list(node_links)[:max_visit_in_every_depth]

            for link in node_links:
                dst = link[1]
                if not visited[dst]:
                    queue.append((dst, depth + 1))
                    visited[dst] = True
        solution['num_attempt_times'] = num_attempt_times
        return solution

    def find_shortest_paths(self, v_net, p_net, v_link, p_pair, method='k_shortest', k=10, max_hop=1e6):
        """
        Optional methods: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        source, target = p_pair
        assert method in ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']

        # Get Latency Attribute
        if self.link_latency_attrs:
            weight = self.link_latency_attrs[0].name
        else:
            weight = None

        try:
            # these three methods do not check any link constraints
            if method == 'first_shortest':
                shortest_paths = [nx.dijkstra_path(p_net, source, target, weight=weight)]
            elif method == 'k_shortest':
                shortest_paths = list(islice(nx.shortest_simple_paths(p_net, source, target, weight=weight), k))
            elif method == 'all_shortest':
                shortest_paths = list(nx.all_shortest_paths(p_net, source, target, weight=weight))
            # these two methods return a fessible path or empty by considering link constraints
            elif method == 'bfs_shortest':
                if weight is not None:
                    raise NotImplementedError('BFS Shortest Path Method not supports seeking for weighted shorest path!')
                shortest_path = self.find_bfs_shortest_path(v_net, p_net, v_link, source, target, weight=weight)
                shortest_paths = [] if shortest_path is None else [shortest_path]
            elif method == 'available_shortest':
                temp_p_net = self.create_available_network(v_net, p_net, v_link)
                shortest_paths = [nx.dijkstra_path(temp_p_net, source, target, weight=weight)]
            elif method == 'available_k_shortest':
                temp_p_net = self.create_available_network(v_net, p_net, v_link)
                shortest_paths = list(islice(nx.shortest_simple_paths(p_net, source, target, weight=weight), k))
        except NotImplementedError as e:
            print(e)
        except Exception as e:
            shortest_paths = []
        if len(shortest_paths) and len(shortest_paths[0]) > max_hop: 
            shortest_paths = []
        return shortest_paths

    def create_available_network(self, v_net, p_net, v_link_pair):
        def available_link(n1, n2):
            p_link = p_net.links[(n1, n2)]
            result, info = self.check_link_constraints(v_net, p_net, v_link, p_link)
            return result
        v_link = v_net.links[v_link_pair]
        sub_graph = nx.subgraph_view(p_net, filter_edge=available_link)
        return sub_graph

    def create_pruned_network(self, v_net, p_net, v_link_pair, ratio=1., div=0.):
        """
        In Proc. IEEE ICCSE, 2016
        A virtual network embedding algorithm based on the connectivity of residual substrate network
        """
        def available_link(n1, n2):
            p_link = p_net.links[(n1, n2)]
            result, info = self.check_attributes(v_link, p_link, e_attr_list)
            return result
        v_link = copy.deepcopy(v_net.links[v_link_pair])
        e_attr_list = self.link_resource_attrs
        for l_attr in e_attr_list:
            v_link[l_attr.name] *= ratio
            v_link[l_attr.name] -= div
        sub_graph = nx.subgraph_view(p_net, filter_edge=available_link)
        return sub_graph

    def route_v_links_with_mcf(self, v_net, p_net, v_link_list, solution):
        if len(v_link_list) == 0:
            return True, {}
        def get_link_bandiwidth(p_link):
            return p_net.links[p_link][self.link_resource_attrs[0].name]
        def get_link_weight(p_link):
            return 1

        p_pair_list, p_pair2v_link, request_dict = list(), dict(), dict()
        for v_link in v_link_list:
            p_pair = (solution['node_slots'][v_link[0]], solution['node_slots'][v_link[1]])
            p_pair_list.append(p_pair)
            p_pair2v_link[p_pair] = v_link
            request_dict[p_pair] = v_net.links[v_link][self.link_resource_attrs[0].name]
        # declare solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        # set parameters
        # solver.set_time_limit(600)
        # define varibles
        directed_p_graph = p_net.to_directed()
        f = {}
        for p_pair in p_pair_list:
            f[p_pair] = {}
            for p_link in directed_p_graph.edges():
                f[p_pair][p_link] = solver.IntVar(0., 9999, f'{p_pair}-{p_link}')
        # set objective
        solver.Minimize(
            sum(get_link_weight(p_link) * sum(f[p_pair][p_link] for p_pair in p_pair_list) 
            for p_link in directed_p_graph.edges)
        )
        # set constraints
        # (1) constraints related to edges
        for p_link in directed_p_graph.edges:
            # if p_link[0] > p_link[1]:
                # continue
            flow_p_pair_p_link = []
            for p_pair in p_pair_list:
                flow_p_pair_p_link.append(f[p_pair][p_link])
                flow_p_pair_p_link.append(f[p_pair][p_link[1], p_link[0]])
            solver.Add(sum(flow_p_pair_p_link) - get_link_bandiwidth(p_link) <= 0)
        # (2) constraints related to nodes
        # for node in directed_p_graph.nodes:
            # if node not in [p_pair[0] for p_pair in p_pair_list] + [p_pair[1] for p_pair in p_pair_list]:
            #     for p_pair in p_pair_list:
            #         solver.Add( 
            #             sum(f[p_pair][p_link] for p_link in directed_p_graph.out_edges(nbunch=node)) - \
            #             sum(f[p_pair][p_link] for p_link in directed_p_graph.in_edges(nbunch=node))== 0)
        # (3) constraints related to source and target nodes
        for p_pair in p_pair_list:
            source_node, target_node = p_pair
            solver.Add(sum(f[p_pair][p_link] for p_link in directed_p_graph.out_edges(nbunch=source_node)) == request_dict[p_pair])
            solver.Add(sum(f[p_pair][p_link] for p_link in directed_p_graph.in_edges(nbunch=source_node)) == 0)
            solver.Add(sum(f[p_pair][p_link] for p_link in directed_p_graph.in_edges(nbunch=target_node)) == request_dict[p_pair])
            solver.Add(sum(f[p_pair][p_link] for p_link in directed_p_graph.out_edges(nbunch=target_node)) == 0)
            for node in directed_p_graph.nodes:
                if node in [source_node, target_node]:
                    continue
                solver.Add( 
                    sum(f[p_pair][p_link] for p_link in directed_p_graph.out_edges(nbunch=node)) - \
                    sum(f[p_pair][p_link] for p_link in directed_p_graph.in_edges(nbunch=node))== 0)

        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # print('Objective value =', solver.Objective().Value())
            for e in p_net.links:
                s_1 = sum([f[p_pair][e].solution_value() for p_pair in p_pair_list])
                s_2 = sum([f[p_pair][e[1], e[0]].solution_value() for p_pair in p_pair_list])
                assert s_1 + s_2 <= p_net.links[e]['bw']

            for p_pair in p_pair_list:
                v_link = p_pair2v_link[p_pair]
                total_link_resource = request_dict[p_pair]
                if total_link_resource == 0:
                    shortest_path = nx.shortest_path(p_net, p_pair[0], p_pair[1])
                    p_links = self.path_to_links(shortest_path)
                    for p_link in p_links:
                        used_link_resources = {self.link_resource_attrs[0].name: 0}
                        solution['link_paths_info'][(v_link, p_link)] = used_link_resources
                else:
                    p_links = []
                    for p_link in directed_p_graph.edges():
                        if f[p_pair][p_link].solution_value():
                            used_link_resources = {
                                self.link_resource_attrs[0].name: f[p_pair][p_link].solution_value() 
                            }
                            solution['link_paths_info'][(v_link, p_link)] = used_link_resources
                            p_links.append(p_link)
                            self.update_link_resources(p_net, p_link, used_link_resources, operator='-')
                    assert len(p_links) > 0
                    # check
                    G = nx.DiGraph()
                    G.add_edges_from(p_links)
                    all_shortest_paths = list(nx.shortest_simple_paths(G, p_pair[0], p_pair[1]))
                    source_p_links = set([(path[0], path[1]) for path in all_shortest_paths])
                    target_p_links = set([(path[-2], path[-1]) for path in all_shortest_paths])
                    assert sum([f[p_pair][source_p_link].solution_value() for source_p_link in source_p_links]) == total_link_resource
                    assert sum([f[p_pair][target_p_link].solution_value() for target_p_link in target_p_links]) == total_link_resource

                assert sum((p_link[1], p_link[0]) in p_links for p_link in p_links) == 0
                solution['link_paths'][v_link] = p_links
            return True, {}
        else:
            return False, {}
        
    def find_bfs_shortest_path(self, v_net, p_net, v_link, source, target, weight=None):
        visit_states = [0] * p_net.num_nodes
        predecessors = {p_n_id: None for p_n_id in range(p_net.num_nodes)}
        Q = deque()
        Q.append((source, []))
        found_target = False
        while len(Q) and not found_target:
            current_node, current_path = Q.popleft()
            current_path.append(current_node)
            for neighbor in nx.neighbors(p_net, current_node):
                check_result, check_info = self.check_link_constraints(v_net, p_net, v_link, (current_node, neighbor))
                if check_result:
                    temp_current_path = copy.deepcopy(current_path)
                    # found
                    if neighbor == target:
                        found_target = True
                        temp_current_path.append(neighbor)
                        shortest_path = temp_current_path
                        break
                    # unvisited
                    if not visit_states[neighbor]:
                        visit_states[neighbor] = 1
                        Q.append((neighbor, temp_current_path))

        if len(Q) and not found_target:
            return None
        else:
            return shortest_path

    def find_candidate_nodes(self, v_net, p_net, v_node_id, filter=[], check_node_constraint=True, check_link_constraint=True):
        r"""Find candicate nodes according to the restrictions and filter.

        Returns:
            candicate_nodes (list)
        """
        if check_node_constraint:
            checked_nodes = list(p_net.nodes)
            suitable_nodes = []
            for p_node_id in checked_nodes:
                check_result, check_info = self.check_node_constraints(v_net, p_net, v_node_id, p_node_id)
                if check_result:
                    suitable_nodes.append(p_node_id)
            candidate_nodes_with_node_constraint = list(set(suitable_nodes).difference(set(filter)))

        if check_link_constraint:
            aggr_method = 'sum' if self.shortest_method == 'mcf' else 'max'
            checked_nodes = candidate_nodes_with_node_constraint if check_node_constraint else list(p_net.nodes)
            v_link_aggr_resource = np.array(v_net.get_aggregation_attrs_data(self.link_resource_attrs, aggr=aggr_method))
            p_link_aggr_resource = np.array(p_net.get_aggregation_attrs_data(self.link_resource_attrs, aggr=aggr_method))
            new_filter = []
            for p_node_id in checked_nodes:
                for e_attr_id in range(p_link_aggr_resource.shape[0]):
                    if not (v_link_aggr_resource[e_attr_id][v_node_id] <= p_link_aggr_resource[e_attr_id][p_node_id]):
                        new_filter.append(p_node_id)
                        break
            candidate_nodes_with_link_constraint = list(set(checked_nodes).difference(set(new_filter)))
        # if check_link_constraint and len(candidate_nodes_with_node_constraint) != len(candidate_nodes_with_link_constraint):
            # print(f'Checking link constraints is helpful: {len(candidate_nodes_with_node_constraint)-len(candidate_nodes_with_link_constraint)}!')
        
        candidate_nodes = candidate_nodes_with_link_constraint if check_link_constraint else candidate_nodes_with_node_constraint
        return candidate_nodes

    def find_feasible_nodes(self, v_net, p_net, v_node_id, node_slots):
        node_constraints_feasible_nodes = []
        for p_node_id in p_net.nodes:
            check_result, check_info = self.check_node_constraints(v_net, p_net, v_node_id, p_node_id)
            if check_result:
                node_constraints_feasible_nodes.append(p_node_id)
        node_constraints_feasible_nodes = list(set(node_constraints_feasible_nodes).difference(set(list(node_slots.values()))))
        feasible_nodes = copy.deepcopy(node_constraints_feasible_nodes)
        for v_neighbor_id, p_neighbor_id in node_slots.items():
            if v_neighbor_id not in v_net.adj[v_node_id]:
                continue
            temp_p_net = self.create_available_network(v_net, p_net, (v_neighbor_id, v_node_id))
            new_feasible_nodes = []
            for p_node_id in feasible_nodes:
                try:
                    shortest_paths = [nx.dijkstra_path(temp_p_net, p_neighbor_id, p_node_id)]
                    new_feasible_nodes.append(p_node_id)
                except:
                    shortest_paths = [[]]
                    # feasible_nodes.remove(p_node_id)
            feasible_nodes = new_feasible_nodes
        return feasible_nodes

    def construct_candidates_dict(self, v_net, p_net):
        candidates_dict = {}
        for v_node_id in list(v_net.nodes):
            candidate_nodes = self.find_candidate_nodes(v_net, p_net, v_node_id)
            candidates_dict[v_node_id] = candidate_nodes
        return candidates_dict

    def node_mapping(self, v_net, p_net, sorted_v_nodes, sorted_p_nodes, solution, reusable=False, inplace=True, matching_mathod='greedy'):
        """
        matching_mathod: ['l2s2', 'greedy']
        """
        assert matching_mathod in ['l2s2', 'greedy']
        solution['node_slots'] = {}
        solution['node_slots_info'] = {}

        p_net = p_net if inplace else copy.deepcopy(p_net)
        sorted_p_nodes = copy.deepcopy(sorted_p_nodes)
        for v_node_id in sorted_v_nodes:
            for p_node_id in sorted_p_nodes:
                place_result, place_info = self.place(v_net, p_net, v_node_id, p_node_id, solution)
                if place_result:
                    if reusable == False: sorted_p_nodes.remove(p_node_id)
                    break
                else:
                    if matching_mathod == 'l2s2':
                        # FAILURE
                        solution.update({'place_result': False, 'result': False})
                        return False
            if not place_result:
                # FAILURE
                solution.update({'place_result': False, 'result': False})
                return False
                
        # SUCCESS
        assert len(solution['node_slots']) == v_net.num_nodes
        return True

    def link_mapping(self, v_net, p_net, solution, sorted_v_links=None, shortest_method='bfs_shortest', k=10, inplace=True):
        """

            shortest_method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        solution['link_paths'] = {}
        solution['link_paths_info'] = {}

        p_net = p_net if inplace else copy.deepcopy(p_net)
        sorted_v_links = sorted_v_links if sorted_v_links is not None else list(v_net.links)
        node_slots = solution['node_slots']

        if shortest_method == 'mcf':
            route_result, route_info = self.route_v_links_with_mcf(v_net, p_net, sorted_v_links, solution)
            if not route_result:
                # FAILURE
                solution.update({'route_result': False, 'result': False})
                return False
        else:
            for v_link in sorted_v_links:
                p_pair = (node_slots[v_link[0]], node_slots[v_link[1]])
                route_result, route_info = self.route(v_net, p_net, v_link, p_pair, solution, shortest_method=shortest_method, k=k)
                if not route_result:
                    # FAILURE
                    solution.update({'route_result': False, 'result': False})
                    return False

        # SUCCESS
        assert len(solution['link_paths']) == v_net.num_links, f"Number of total links: {v_net.num_links}, Number of routed links: {len(solution['link_paths'])}"
        return True

    def unsafely_link_mapping(self, v_net, p_net, solution, sorted_v_links=None, shortest_method='k_shortest', k=10, inplace=True, pruning_ratio=None):
        """
            shortest_method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        p_net = p_net if inplace else copy.deepcopy(p_net)
        sorted_v_links = sorted_v_links if sorted_v_links is not None else list(v_net.links)
        node_slots = solution['node_slots']
        route_check_info_dict = {}
        violation_value_dict = {}
        sum_violation_value = 0
        for v_link_pair in sorted_v_links:
            p_path_pair = (node_slots[v_link_pair[0]], node_slots[v_link_pair[1]])
            route_result, route_check_info, violation_value = self.unsafely_route(v_net, p_net, v_link_pair, p_path_pair, solution, shortest_method=shortest_method, k=k, pruning_ratio=pruning_ratio)
            sum_violation_value += violation_value
            route_check_info_dict[v_link_pair] = route_check_info_dict
            violation_value_dict[v_link_pair] = violation_value
            if not route_result:
                # FAILURE
                solution.update({'route_result': False, 'result': False})
                return False, route_check_info_dict, 0
        # SUCCESS
        assert len(solution['link_paths']) == v_net.num_links
        solution['violation'] = sum_violation_value
        return True, route_check_info_dict, sum_violation_value

    def release(self, v_net, p_net, solution):
        r"""Release occupied resources when a VN leaves p_net."""
        if not solution['result']:
            return False
        for v_node_id, p_node_id in solution['node_slots'].items():
            used_node_resources = solution['node_slots_info'][(v_node_id, p_node_id)]
            self.update_node_resources(p_net, p_node_id, used_node_resources, operator='+')
        for v_link, p_links in solution['link_paths'].items():
            for p_link in p_links:
                used_link_resources = solution['link_paths_info'][(v_link, p_link)]
                self.update_link_resources(p_net, p_link, used_link_resources, operator='+')
        return True

    def deploy_with_node_slots(self, v_net, p_net, node_slots, solution, inplace=True, shortest_method='bfs_shortest', k_shortest=10):
        p_net = p_net if inplace else copy.deepcopy(p_net)
        # unfeasible solution
        if len(node_slots) != v_net.num_nodes or -1 in node_slots:
            solution.update({'place_result': False, 'result': False})
            return
        # node mapping
        node_mapping_result = self.node_mapping(v_net, p_net, list(node_slots.keys()), list(node_slots.values()), solution, 
                                                            reusable=False, inplace=True, matching_mathod='l2s2')
        if not node_mapping_result:
            solution.update({'place_result': False, 'result': False})
            return
        # link mapping
        link_mapping_result = self.link_mapping(v_net, p_net, solution, sorted_v_links=None,
                                                            shortest_method=shortest_method, k=k_shortest, inplace=True)
        
        if not link_mapping_result:
            solution.update({'route_result': False, 'result': False})
            return
        # Success
        solution['result'] = True
        self.counter.count_solution(v_net, solution)
        return

    def unsafely_deploy_with_node_slots(self, v_net, p_net, node_slots, solution, inplace=True, shortest_method='bfs_shortest', k_shortest=10, pruning_ratio=None):
        p_net = p_net if inplace else copy.deepcopy(p_net)
        # unfeasible solution
        if len(node_slots) != v_net.num_nodes or -1 in node_slots:
            solution.update({'place_result': False, 'result': False})
            return
        # node mapping
        node_mapping_result = self.node_mapping(v_net, p_net, 
                                                list(node_slots.keys()), 
                                                list(node_slots.values()), 
                                                solution,
                                                reusable=False, 
                                                inplace=False, 
                                                matching_mathod='l2s2')
        if not node_mapping_result:
            solution.update({'place_result': False, 'result': False})
            return
        # link mapping
        link_mapping_result, route_check_info, violation_value  = self.unsafely_link_mapping(v_net, p_net, 
                                                                                            solution, 
                                                                                            sorted_v_links=None,
                                                                                            shortest_method=shortest_method, 
                                                                                            k=k_shortest, 
                                                                                            inplace=False, 
                                                                                            pruning_ratio=pruning_ratio)
        if not link_mapping_result:
            solution.update({'route_result': False, 'result': False})
            return 
        # Success
        solution['result'] = True
        self.counter.count_solution(v_net, solution)
        return

    def deploy(self, v_net, p_net, solution):
        if not solution['result']:
            return False
        for (v_node_id, p_node_id), used_node_resources in solution['node_slots_info'].items():
            self.update_node_resources(p_net, p_node_id, used_node_resources, operator='-')
        for (v_link, p_link), used_link_resources in solution['link_paths_info'].items():
            self.update_link_resources(p_net, p_link, used_link_resources, operator='-')
        return True

    def path_to_links(self, path):
        assert len(path) > 1
        return [(path[i], path[i+1]) for i in range(len(path)-1)]

if __name__ == '__main__':
    pass
