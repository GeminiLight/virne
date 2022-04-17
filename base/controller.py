from collections import deque
import copy
import numpy as np
import networkx as nx
from itertools import islice

from .recorder import Solution, Counter
from data.physical_network import PhysicalNetwork

'''
Methods List

# 
check_node_constraints
check_link_constraints
check_path_constraints
update_node_resources
update_link_resources
update_path_resources

##
place
route
place_and_route
undo_place
undo_route
undo_place_and_route

##
node_mapping
link_mapping

deploy
release

bfs_deploy
undo_deploy: release and reset the solution




'''

class Controller:
    
    @classmethod
    def check_node_constraints(cls, vn, pn, v_node_id, p_node_id):
        assert p_node_id in list(pn.nodes)
        for n_attr in vn.get_node_attrs():
            if not n_attr.check(vn, pn, v_node_id, p_node_id):
                return False
        return True

    @classmethod
    def check_link_constraints(cls, vn, pn, v_link, p_link):
        for e_attr in vn.edge_attrs:
            if not e_attr.check(vn, pn, v_link, p_link):
                return False
        return True

    @classmethod
    def check_path_constraints(cls, vn, pn, v_link, p_path):
        p_edges = cls.path_to_edges(p_path)
        for p_edge in p_edges:
            result = cls.check_link_constraints(vn, pn, v_link, p_edge)
            if not result:
                return False
        # TODO: Add path constraint
        return True

    @classmethod
    def update_node_resources(cls, vn, pn, v_node_id, p_node_id, operator='-'):
        for n_attr in vn.get_node_attrs('resource'):
            n_attr.update(vn.nodes[v_node_id], pn.nodes[p_node_id], operator)
    
    @classmethod
    def update_link_resources(cls, vn, pn, v_link, p_link, operator='-'):
        for e_attr in vn.edge_attrs:
            e_attr.update_path(vn.edges[v_link], pn, p_link, operator)

    @classmethod
    def update_path_resources(cls, vn, pn, v_link, p_path, operator='-'):
        for e_attr in vn.edge_attrs:
            e_attr.update_path(vn.edges[v_link], pn, p_path, operator)

    @classmethod
    def place(cls, vn, pn, v_node_id, p_node_id, solution=None):
        r"""Attempt to place the VNF `vid` in PN node `pid`."""
        if not cls.check_node_constraints(vn, pn, v_node_id, p_node_id):
            return False
        cls.update_node_resources(vn, pn, v_node_id, p_node_id, operator='-')
        if solution is not None: solution['node_slots'][v_node_id] = p_node_id
        return True

    @classmethod
    def route(cls, vn, pn, v_link, pl_pair, solution=None, shortest_method='all_shortest', k=1):
        r"""Return True if route successfully the virtual link (vid_a, v_node_id_b) in the physical network path (pid_a, p_node_id_b); Otherwise False.
        
            shortest_method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        # place the prev VNF and curr VNF on the identical physical node
        if pl_pair[0] == pl_pair[1]:
            raise NotImplementedError
            return True

        shortest_paths = cls.find_shortest_paths(vn, pn, v_link, pl_pair, method=shortest_method, k=k)
        for p_path in shortest_paths:
            if cls.check_path_constraints(vn, pn, v_link, p_path):
                cls.update_path_resources(vn, pn, v_link, p_path, operator='-')
                if solution is not None: solution['edge_paths'][v_link] = p_path
                return True
        return False

    @classmethod
    def place_and_route(cls, vn, pn, v_node_id, p_node_id, solution, shortest_method='bfs_shortest', k=1) -> bool:
        r"""Attempt to place the VNF `vid` in PN node`pid` 
            and route VLs related to the VNF.

            shortest_method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        # Place
        place_result = cls.place(vn, pn, v_node_id, p_node_id, solution)
        if not place_result:
            solution.update({'place_result': False, 'result': False})
            return False
        # Route
        v_node_id_neighbors = list(vn.adj[v_node_id])
        for n_v_node_id in v_node_id_neighbors:
            placed = n_v_node_id in solution['node_slots'].keys()
            routed = (n_v_node_id, v_node_id) in solution['edge_paths'].keys() or (v_node_id, n_v_node_id) in solution['edge_paths'].keys()
            if placed and not routed:
                n_p_node_id = solution['node_slots'][n_v_node_id]
                route_result = cls.route(vn, pn, (v_node_id, n_v_node_id), (p_node_id, n_p_node_id), solution, 
                                            shortest_method=shortest_method, k=k)
                if not route_result:
                    solution.update({'route_result': False, 'result': False})
                    return False
        return True

    @classmethod
    def undo_place(cls, vn, pn, v_node_id, p_node_id, solution=None):
        cls.update_node_resources(vn, pn, v_node_id, p_node_id, operator='+')
        if solution is not None: del solution['node_slots'][v_node_id]
        return True

    @classmethod
    def undo_route(cls, vn, pn, v_link, p_path, solution=None):
        cls.update_path_resources(vn, pn, v_link, p_path, operator='+')
        if solution is not None: del solution['edge_paths'][v_link]
        return True

    @classmethod
    def undo_place_and_route(cls, vn, pn, v_node_id, p_node_id, solution):
        # Undo place
        origin_node_slots = list(solution['node_slots'].keys())
        if v_node_id not in origin_node_slots:
            return True
        undo_place_result = cls.undo_place(vn, pn, v_node_id, p_node_id, solution)
        # Undo route
        origin_node_slots = list(solution['edge_paths'].keys())
        for v_link in origin_node_slots:
            if v_node_id in v_link:
                undo_route_result = cls.undo_route(vn, pn, v_link, solution['edge_paths'][v_link], solution)
        return True

    @classmethod
    def undo_deploy(cls, vn, pn, solution):
        r"""Release occupied resources when a VN leaves PN, and reset the solution."""
        cls.release(vn, pn, solution)
        solution.reset()
        return True

    @classmethod
    def bfs_deploy(cls, vn, pn, sorted_v_nodes, pn_initial_node, max_visit=100, max_depth=10, shortest_method='all_shortest', k=10):
        r"""Deploy the `vn` in `pn` starting from `initial_node` using Breadth-First Search solverrithm.

        method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        solution = Solution(vn)

        max_visit_in_every_depth = int(np.power(max_visit, 1 / max_depth))
        
        curr_depth = 0
        visited = pn.num_nodes * [False]
        queue = [(pn_initial_node, curr_depth)]
        visited[pn_initial_node] = True

        num_placed_nodes = 0
        v_node_id = sorted_v_nodes[num_placed_nodes]

        while queue:
            (curr_pid, depth) = queue.pop(0)
            if depth > max_depth:
                break

            if cls.place_and_route(vn, pn, v_node_id, curr_pid, solution, shortest_method=shortest_method, k=k):
                num_placed_nodes = num_placed_nodes + 1

                if num_placed_nodes >= len(sorted_v_nodes):
                    solution['result'] = True
                    return solution
                v_node_id = sorted_v_nodes[num_placed_nodes]
            else:
                cls.undo_place_and_route(vn, pn, v_node_id, curr_pid, solution)

            if depth == max_depth:
                continue

            node_edges = pn.edges(curr_pid, data=True)
            node_edges = node_edges if len(node_edges) <= max_visit else node_edges[:max_visit_in_every_depth]

            for edge in node_edges:
                dst = edge[1]
                if not visited[dst]:
                    queue.append((dst, depth + 1))
                    visited[dst] = True
        return solution

    @classmethod
    def find_shortest_paths(cls, vn, pn, v_link, p_pair, method='k_shortest', k=10):
        """
        Optional methods: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        """
        source, target = p_pair
        assert method in ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
        try:
            # these three methods do not check any link constraints
            if method == 'first_shortest':
                shortest_path = [nx.dijkstra_path(pn, source, target)]
                return shortest_path
            elif method == 'k_shortest':
                return list(islice(nx.shortest_simple_paths(pn, source, target), k))
            elif method == 'all_shortest':
                return list(nx.all_shortest_paths(pn, source, target))
            # these two methods return a fessible path or empty by considering link constraints
            elif method == 'bfs_shortest':
                return [cls.find_bfs_shortest_path(vn, pn, v_link, source, target)]
            elif method == 'available_shortest':
                temp_pn = cls.create_available_network(vn, pn, v_link)
                shortest_path = [nx.dijkstra_path(temp_pn, source, target)]
                return shortest_path
        except:
            return []

    @classmethod
    def create_available_network(cls, vn, pn, v_link):
        def available_edge(n1, n2):
            return cls.check_link_constraints(vn, pn, v_link, (n1, n2))
        sub_graph = nx.subgraph_view(pn, filter_edge=available_edge)
        return sub_graph

    @classmethod
    def find_bfs_shortest_path(cls, vn, pn, v_link, source, target):
        visit_states = [0] * pn.num_nodes
        predecessors = {p_n_id: None for p_n_id in range(pn.num_nodes)}
        Q = deque()
        Q.append((source, []))
        found_target = False
        while len(Q) and not found_target:
            current_node, current_path = Q.popleft()
            current_path.append(current_node)
            for neighbor in nx.neighbors(pn, current_node):
                if cls.check_link_constraints(vn, pn, v_link, (current_node, neighbor)):
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
            return []
        else:
            return shortest_path
        if predecessors[target] is None:
            return []
        path = [target]
        current_node = predecessors[target]
        while current_node != source:
            path.append(current_node)
            current_node = predecessors[current_node]
        path.append(current_node)
        path.reverse()
        return path

    @classmethod
    def find_candidate_nodes(cls, pn, vn, v_node_id, filter=[], check_edge_constraint=False):
        r"""Find candicate nodes according to the restrictions and filter.

        Returns:
            candicate_nodes (list)
        """
        suitable_nodes = []
        for p_node_id in pn.nodes:
            if cls.check_node_constraints(vn, pn, v_node_id, p_node_id):
                suitable_nodes.append(p_node_id)
        candidate_nodes = list(set(suitable_nodes).difference(set(filter)))
        return candidate_nodes

    @classmethod
    def construct_candidates_dict(self, vn, pn):
        candidates_dict = {}
        for v_node_id in range(vn.num_nodes):
            candidate_nodes = self.find_candidate_nodes(pn, vn, v_node_id)
            candidates_dict[v_node_id] = candidate_nodes
        return candidates_dict

    @classmethod
    def node_mapping(cls, vn, pn, sorted_v_nodes, sorted_p_nodes, solution, reusable=False, inplace=True, matching_mathod='greedy'):
        """
        matching_mathod: ['l2s2', 'greedy']
        """
        assert matching_mathod in ['l2s2', 'greedy']
        pn = pn if inplace else copy.deepcopy(pn)
        sorted_p_nodes = copy.deepcopy(sorted_p_nodes)
        for v_node_id in sorted_v_nodes:
            for p_node_id in sorted_p_nodes:
                place_result = Controller.place(vn, pn, v_node_id, p_node_id, solution)
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
        assert len(solution['node_slots']) == vn.num_nodes
        return True

    @classmethod
    def link_mapping(cls, vn, pn, solution, sorted_v_edges=None, shortest_method='bfs_shortest', k=10, inplace=True):
        """
        Optional shortest_methods: ['first_shortest', 'all_shortest', 'k_shortest' with 'k', 'bfs_shortest']
        """
        pn = pn if inplace else copy.deepcopy(pn)
        sorted_v_edges = sorted_v_edges if sorted_v_edges is not None else list(vn.edges)
        node_slots = solution['node_slots']

        for v_link in sorted_v_edges:
            p_pair = (node_slots[v_link[0]], node_slots[v_link[1]])
            route_result = Controller.route(vn, pn, v_link, p_pair, solution, shortest_method=shortest_method, k=k)
            if not route_result:
                # FAILURE
                solution.update({'route_result': False, 'result': False})
                return False
        # SUCCESS
        assert len(solution['edge_paths']) == vn.num_edges
        return True

    @classmethod
    def release(cls, vn, pn, solution):
        r"""Release occupied resources when a VN leaves PN."""
        if not solution['result']:
            return False
        for v_node_id, p_node_id in solution['node_slots'].items():
            cls.update_node_resources(vn, pn, v_node_id, p_node_id, operator='+')
        for v_link, p_path in solution['edge_paths'].items():
            cls.update_path_resources(vn, pn, v_link, p_path, operator='+')
        return True

    def deploy_with_node_slots(self, vn, pn, node_slots, solution, inplace=True):
        pn = pn if inplace else copy.deepcopy(pn)
        # unfeasible solution
        if len(node_slots) != vn.num_nodes or -1 in node_slots:
            solution.update({'place_result': False, 'result': False})
            return
        # node mapping
        node_mapping_result = Controller.node_mapping(vn, pn, list(node_slots.keys()), list(node_slots.values()), solution, 
                                                            reusable=False, inplace=True, matching_mathod='l2s2')
        if not node_mapping_result:
            solution.update({'place_result': False, 'result': False})
            return
        # link mapping
        link_mapping_result = Controller.link_mapping(vn, pn, solution, sorted_v_edges=None,
                                                            shortest_method='bfs_shortest', k=1, inplace=True)
        if not link_mapping_result:
            solution.update({'route_result': False, 'result': False})
            return 
        # Success
        solution['result'] = True
        Counter.count_solution(vn, solution)
        return

    @classmethod
    def deploy(cls, vn, pn, solution):
        if not solution['result']:
            return False
        for v_node_id, p_node_id in solution['node_slots'].items():
            cls.update_node_resources(vn, pn, v_node_id, p_node_id, operator='-')
        for v_link, p_path in solution['edge_paths'].items():
            cls.update_path_resources(vn, pn, v_link, p_path, operator='-')
        return True

    @staticmethod
    def path_to_edges(path):
        assert len(path) > 1
        return [(path[i], path[i+1]) for i in range(len(path)-1)]

if __name__ == '__main__':
    pass
