# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
from typing import *
import numpy as np
import networkx as nx
from itertools import islice
from collections import deque

from .solution import Solution
from virne.utils import flatten_recurrent_dict, path_to_links
from virne.data.network import Network
from virne.data.physical_network import PhysicalNetwork
from virne.data.virtual_network import VirtualNetwork
from virne.data.attribute import create_attrs_from_setting


class Controller:
    """
    A class that controls changes in the physical network, i.e., execute the resource allocation process

    Attributes:
        all_node_attrs (list): A list of all node attributes.
        all_link_attrs (list): A list of all link attributes.
        node_resource_attrs (list): A list of node resource attributes.
        link_resource_attrs (list): A list of link resource attributes.
        link_latency_attrs (list): A list of link latency attributes.
        reusable (bool): A boolean indicating if the resources can be reused.
        node_ranking_method (str): A string indicating the ranking method for the nodes.
        link_ranking_method (str): A string indicating the ranking method for the links.
        matching_mathod (str): A string indicating the matching method.
        shortest_method (str): A string indicating the shortest path method.

    Methods:
        ### --- Check constraints --- ### 
        check_attributes: Check the attributes.
        check_node_constraints: Check the node constraints.
        check_link_constraints: Check the link constraints.
        check_path_constraints: Check the path constraints.
        check_graph_constraints: Check the graph constraints.

        ### --- Update resources --- ###
        update_node_resources: Update the node resources.
        update_link_resources: Update the link resources.
        update_path_resources: Update the path resources.

        ### --- Place --- ###
        place: Place the virtual network.
        safely_place: Place the virtual network, ensuring that the constraints are satisfied.
        unsafely_place: Place the virtual network without checking the constraints.
        undo_place: Undo the placement.

        ### --- Route --- ###
        route: Route the virtual network.
        safely_route: Route the virtual network, ensuring that the constraints are satisfied.
        unsafely_route: Route the virtual network without checking the constraints.
        undo_route: Undo the routing.

        ### --- Place and route --- ###
        place_and_route: Place and route the virtual network.
        safely_place_and_route: Place and route the virtual network, ensuring that the constraints are satisfied.
        unsafely_place_and_route: Place and route the virtual network without checking the constraints.
        undo_place_and_route: Undo the placement and routing.

        ### --- Node Mapping --- ###
        node_mapping: Map the virtual nodes to the physical nodes.
        safely_node_mapping: Map the virtual nodes to the physical nodes, ensuring that the constraints are satisfied.
        unsafely_node_mapping: Map the virtual nodes to the physical nodes without checking the constraints.

        ### --- Link Mapping --- ###
        link_mapping: Map the virtual links to the physical links.
        safely_link_mapping: Map the virtual links to the physical links, ensuring that the constraints are satisfied.
        unsafely_link_mapping: Map the virtual links to the physical links without checking the constraints.

        ### --- Deploy --- ###
        deploy: Deploy the virtual network.
        safely_deploy: Deploy the virtual network, ensuring that the constraints are satisfied.
        unsafely_deploy: Deploy the virtual network without checking the constraints.
        undo_deploy: Undo the deployment.
        release: Release the virtual network.

        deploy_with_node_slots: Deploy the virtual network with node slots.
        safely_deploy_with_node_slots: Deploy the virtual network with node slots, ensuring that the constraints are satisfied.
        unsafely_deploy_with_node_slots: Deploy the virtual network with node slots without checking the constraints.

        bfs_deploy: Deploy the virtual network using BFS.
    """    

    def __init__(self, node_attrs_setting : list=[], link_attrs_setting : list=[], **kwargs) -> None:
        """
        Initializes a general controller.

        Args:
            node_attrs_setting (list): A list of node attributes settings.
            link_attrs_setting (list): A list of link attributes settings.
            **kwargs: Optional keyword arguments:
                reusable (bool): A boolean indicating if the resources can be reused.
                node_ranking_method (str): A string indicating the ranking method for the nodes.
                link_ranking_method (str): A string indicating the ranking method for the links.
                matching_mathod (str): A string indicating the matching method.
                shortest_method (str): A string indicating the shortest path method.
        """
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

    def check_attributes(self, v: dict, p: dict, attrs_list: list) -> Tuple[bool, dict]:
        """
        Check if node-level or link-level specified attributes in the specified node are satisfied.

        Args:
            v (dict): The attributes of one virtual node (link) to be checked.
            p (dict): The attributes of one physical node (link) to be checked.
            attrs_list (list): The list of attributes to be checked.

        Returns:
            final_result (bool): True if all attributes are satisfied, False otherwise.
            satisfiability_info (dict): A dictionary containing the satisfiability information of the attributes.
        """
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

    def check_graph_constraints(self, v_net: VirtualNetwork, p_net: PhysicalNetwork) -> Tuple[bool, dict]:
        """
        Check if the specified graph constraints in the specified networks are satisfied.

        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.

        Returns:
            final_result (bool): True if all graph constraints are satisfied, False otherwise.
            graph_satisfiability_info (dict): A dictionary containing the satisfiability information of the graph constraints.

        """
        final_result, graph_satisfiability_info = self.check_attributes(v_net, p_net, self.all_node_attrs)
        return final_result, graph_satisfiability_info

    def check_node_constraints(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_node_id: int,
            p_node_id: int
        ) -> Tuple[bool, dict]:
        """
        Check if the specified node constraints in the specified networks are satisfied.

        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            v_node_id (int): The virtual node ID.
            p_node_id (int): The physical node ID.

        Returns:
            final_result (bool): True if all node constraints are satisfied, False otherwise.
            node_satisfiability_info (dict): A dictionary containing the satisfiability information of the node constraints.
        """
        assert p_node_id in list(p_net.nodes)
        v_node, p_node = v_net.nodes[v_node_id], p_net.nodes[p_node_id]
        final_result, node_satisfiability_info = self.check_attributes(v_node, p_node, self.all_node_attrs)
        return final_result, node_satisfiability_info

    def check_link_constraints(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_link_pair: Union[list, tuple], 
            p_link_pair: Union[list, tuple]
        ) -> Tuple[bool, dict]:
        """Check if the link-level constraints are satisfied between a virtual link and its mapped physical link.
        
        Args:
            v_net (VirtualNetwork): The virtual network for which the link-level constraints are to be checked.
            p_net (PhysicalNetwork): The physical network for which the link-level constraints are to be checked.
            v_link_pair (Union[list, tuple]): A list or tuple of length 2, representing the ID pair of the virtual link.
            p_link_pair (Union[list, tuple]): A list or tuple of length 2, representing the ID pair of the physical link.

        Returns:
            final_result (bool): A boolean value indicating whether all the link-level constraints are satisfied.
            link_satisfiability_info (dict): A dictionary containing the satisfiability information of the link-level constraints.
                                              The keys of the dictionary are the names of the constraints,
                                              and the values are either the violation values or 0 if the constraint is satisfied.
        """
        v_link, p_link = v_net.links[v_link_pair], p_net.links[p_link_pair]
        final_result, link_satisfiability_info = self.check_attributes(v_link, p_link, self.all_link_attrs)
        return final_result, link_satisfiability_info

    def check_path_latency_constraints(
            self,
            v_net: VirtualNetwork,
            p_net: PhysicalNetwork,
            v_link: dict,
            p_path: list
        ) -> Tuple[bool, dict]:
        """
        Check if the path-level latency constraints are satisfied for a given virtual link and its mapped physical path.

        Args:
            v_net (VirtualNetwork): The virtual network for which the path-level constraints are to be checked.
            p_net (PhysicalNetwork): The physical network for which the path-level constraints are to be checked.
            v_link (dict): A dictionary representing the virtual link.
            p_path (list): A list of nodes representing the physical path.

        Returns:
            final_result (bool): A boolean value indicating whether all the path-level constraints are satisfied.
            path_satisfiability_info (dict): A dictionary containing the satisfiability information of the path-level constraints.
                                             The keys of the dictionary are the IDs of the physical links in the path,
                                             and the values are dictionaries containing the satisfiability information of the
                                             link-level constraints, in the same format as the return value of `check_link_constraints()`.
        """
        p_links = path_to_links(p_path)
        final_result = True
        path_satisfiability_info = dict()
        for p_link in p_links:
            result, info = self.check_link_constraints(v_net, p_net, v_link, p_link)
            if not result:
                final_result = False
            path_satisfiability_info[p_link] = info


    def check_path_constraints(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_link: dict, 
            p_path: list
        ) -> Tuple[bool, dict]:
        """
        Check if the path-level constraints are satisfied for a given virtual link and its mapped physical path.

        Args:
            v_net (VirtualNetwork): The virtual network for which the path-level constraints are to be checked.
            p_net (PhysicalNetwork): The physical network for which the path-level constraints are to be checked.
            v_link (dict): A dictionary representing the virtual link.
            p_path (list): A list of nodes representing the physical path.

        Returns:
            final_result (bool): A boolean value indicating whether all the path-level constraints are satisfied.
            path_satisfiability_info (dict): A dictionary containing the satisfiability information of the path-level constraints.
                                             The keys of the dictionary are the IDs of the physical links in the path,
                                             and the values are dictionaries containing the satisfiability information of the
                                             link-level constraints, in the same format as the return value of `check_link_constraints()`.
        """
        p_links = path_to_links(p_path)
        final_result = True
        path_satisfiability_info = dict()
        for p_link in p_links:
            result, info = self.check_link_constraints(v_net, p_net, v_link, p_link)
            if not result:
                final_result = False
            path_satisfiability_info[p_link] = info
        # TODO: Add path constraint
        return final_result, path_satisfiability_info

    def update_resource(
            self, 
            net: Network, 
            element_owner: str, 
            element_id: int, 
            attr_name: str, 
            value: float, 
            operator: Optional[str] = '-', 
            safe: Optional[bool] = True
        ) -> None:
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

    def update_node_resources(
            self, 
            p_net, 
            p_node_id, 
            used_node_resources, 
            operator: Optional[str] = '-', 
            safe: Optional[bool] = True
        ) -> None:
        for n_attr_name, value in used_node_resources.items():
            self.update_resource(p_net, 'node', p_node_id, n_attr_name, value, operator=operator, safe=safe)

    def update_link_resources(self, p_net, p_link, used_link_resources, operator='-', safe=True):
        for e_attr_name, value in used_link_resources.items():
            self.update_resource(p_net, 'link', p_link, e_attr_name, value, operator=operator, safe=safe)

    def update_path_resources(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, v_link, p_path, operator='-', safe=True):
        for l_attr in self.link_resource_attrs:
            l_attr.update_path(v_net.links[v_link], p_net, p_path, operator, safe=safe)

    def place(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_node_id: int, 
            p_node_id: int, 
            solution: Solution,
            check_feasibility: bool = True
        ) -> Tuple[bool, dict]:
        """
        Attempt to place the virtual node `v_node_id` in the physical node `p_node_id`.
        
        Args:
            v_net (VirtualNetwork): The virtual network for which the placement is to be performed.
            p_net (PhysicalNetwork): The physical network for which the placement is to be performed.
            v_node_id (int): The ID of the virtual node to be placed.
            p_node_id (int): The ID of the physical node on which the virtual node is to be placed.
            solution (Solution): The solution object to which the placement is to be added.
            check_feasibility (bool): A boolean value indicating whether the placement should be checked for feasibility.

        Returns:
            result (bool): A boolean value indicating whether the placement was successful.
            check_info (dict): A dictionary containing the satisfiability information of the node-level constraints.
                                The keys of the dictionary are the names of the node-level constraints,
                                and the values are boolean values indicating whether the constraint is satisfied.
        """
        if check_feasibility:
            return self.safely_place(v_net, p_net, v_node_id, p_node_id, solution)
        else:
            return self.unsafely_place(v_net, p_net, v_node_id, p_node_id, solution)

    def safely_place(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_node_id: int, 
            p_node_id: int, 
            solution: Solution
        ) -> Tuple[bool, dict]:
        """
        Attempt to place the virtual node `v_node_id` in the physical node `p_node_id`, ensuring all constraints are satisfied.
        """
        check_result, check_info = self.check_node_constraints(v_net, p_net, v_node_id, p_node_id)
        if not check_result:
            return False, check_info
        used_node_resources = {n_attr.name: v_net.nodes[v_node_id][n_attr.name] for n_attr in self.node_resource_attrs}
        self.update_node_resources(p_net, p_node_id, used_node_resources, operator='-')
        solution['node_slots'][v_node_id] = p_node_id
        solution['node_slots_info'][(v_node_id, p_node_id)] = used_node_resources
        return True, check_info

    def unsafely_place(
            self, 
            v_net: VirtualNetwork,
            p_net: PhysicalNetwork, 
            v_node_id: int, 
            p_node_id: int, 
            solution: Solution
        ) -> Tuple[bool, dict]:
        """
        Attempt to place the virtual node `v_node_id` in the physical node `p_node_id`, without checking the feasibility of the solution.
        """
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
        solution['v_net_violation'] += violation_value
        solution['v_net_place_violation'] += violation_value
        solution['v_net_current_violation'] = violation_value
        return True, check_info
    
    def undo_place(self, v_node_id: int, p_net: PhysicalNetwork, solution: Solution) -> bool:
        """
        Undo the placement of a virtual node in the given solution.
        
        Args:
            v_node_id (int): The ID of the virtual node.
            p_net (PhysicalNetwork): The physical network object.
            solution (Solution): The solution object.

        Returns:
            bool: Return True if the placement is successfully undone.
        """
        assert v_node_id in solution['node_slots'].keys()
        p_node_id = solution['node_slots'][v_node_id]
        used_node_resources = solution['node_slots_info'][(v_node_id, p_node_id)]
        self.update_node_resources(p_net, p_node_id, used_node_resources, operator='+')
        del solution['node_slots'][v_node_id]
        del solution['node_slots_info'][(v_node_id, p_node_id)]
        return True

    def route(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_link: tuple, 
            pl_pair: tuple, 
            solution: Solution = None, 
            shortest_method: str = 'bfs_shortest',
            k: int = 1, 
            rank_path_func: Callable = None, 
            check_feasibility: bool = True
        ) -> Tuple[bool, dict]:
        """
        Attempt to route the virtual link `v_link` in the physical network path `pl_pair`.
        
        Args:
            v_net (VirtualNetwork): The virtual network for which the routing is to be performed.
            p_net (PhysicalNetwork): The physical network for which the routing is to be performed.
            v_link (tuple): The ID of the virtual link to be routed.
            pl_pair (tuple): The physical link pair on which the virtual link is to be routed.
            solution (Solution): The solution object to which the routing is to be added.
            shortest_method (str): The method used to find the shortest path.
                                    ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
            k (int): The number of shortest paths to be found.
            rank_path_func (Callable): The function used to rank the paths.
            check_feasibility (bool): A boolean value indicating whether the routing should be checked for feasibility.

        Returns:
            result (bool): A boolean value indicating whether the routing was successful.
            check_info (dict): A dictionary containing the satisfiability information of the link-level constraints.
                                The keys of the dictionary are the names of the link-level constraints,
                                and the values are boolean values indicating whether the constraint is satisfied.
        """
        if check_feasibility:
            return self.safely_route(v_net, p_net, v_link, pl_pair, solution, shortest_method, k, rank_path_func)
        else:
            return self.unsafely_route(v_net, p_net, v_link, pl_pair, solution, shortest_method, k, rank_path_func)

    def safely_route(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_link: tuple, 
            pl_pair: tuple, 
            solution: Solution = None, 
            shortest_method: str = 'all_shortest',
            k: int = 1, 
            rank_path_func: Callable = None, 
        ) -> Tuple[bool, dict]:
        """
        Attempt to route the virtual link `v_link` in the physical network path `pl_pair`, ensuring all constraints are satisfied.
        """
        # currently, only first_shortest, k_shortest and all_shortest support unsafe routing mode
        # place the prev VNF and curr VNF on the identical physical node
        if v_link in solution['link_paths']:
            for p_link in solution['link_paths'][v_link]:
                solution['link_paths_info'].pop((v_link, p_link), None)
        solution['link_paths'][v_link] = []

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
                p_links = path_to_links(p_path)
                solution['link_paths'][v_link] = p_links
                for p_link in p_links:
                    used_link_resources = {l_attr.name: v_net.links[v_link][l_attr.name] for l_attr in self.link_resource_attrs}
                    self.update_link_resources(p_net, p_link, used_link_resources, operator='-', safe=True)
                    solution['link_paths_info'][(v_link, p_link)] = used_link_resources
                return True, check_info
        return False, check_info

    def unsafely_route(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_link: tuple, 
            pl_pair: tuple, 
            solution: Solution = None, 
            shortest_method: str = 'all_shortest',
            k: int = 1, 
            rank_path_func: Callable = None, 
            pruning_ratio: float = None
        ) -> Tuple[bool, dict]:
        """
        Attempt to route the virtual link `v_link` in the physical network path `pl_pair`, without checking the feasibility of the solution.
        """
        # place the prev VNF and curr VNF on the identical physical node
        check_info = {l_attr.name: 0. for l_attr in v_net.get_link_attrs()}
        if pl_pair[0] == pl_pair[1]:
            if self.reusable:
                return True, check_info
            else:
                raise NotImplementedError

        if v_link in solution['link_paths']:
            for p_link in solution['link_paths'][v_link]:
                solution['link_paths_info'].pop((v_link, p_link), None)
        solution['link_paths'][v_link] = []

        pruned_p_net = p_net if pruning_ratio is None else self.create_pruned_network(v_net, p_net, v_link_pair=v_link, ratio=pruning_ratio, div=0)
        shortest_paths = self.find_shortest_paths(v_net, pruned_p_net, v_link, pl_pair, method=shortest_method, k=k)
        if len(shortest_paths) == 0:
            solution['v_net_current_violation'] = 0.
            return False, check_info
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
                p_links = path_to_links(p_path)
                solution['link_paths'][v_link] = p_links
                for p_link in p_links:
                    used_link_resources = {l_attr.name: v_net.links[v_link][l_attr.name] for l_attr in self.link_resource_attrs}
                    self.update_link_resources(p_net, p_link, used_link_resources, operator='-', safe=True)
                    solution['link_paths_info'][(v_link, p_link)] = used_link_resources
                    solution['v_net_current_violation'] = 0.
                return True, check_info
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
        p_links = path_to_links(best_p_path)
        solution['link_paths'][v_link] = p_links
        for p_link in p_links:
            used_link_resources = {l_attr.name: v_net.links[v_link][l_attr.name] for l_attr in self.link_resource_attrs}
            self.update_link_resources(p_net, p_link, used_link_resources, operator='-', safe=False)
            solution['link_paths_info'][(v_link, p_link)] = used_link_resources
        solution['v_net_violation'] += best_violation_value
        solution['v_net_route_violation'] += best_violation_value
        solution['v_net_current_violation'] = best_violation_value
        return True, check_info

    def undo_route(self, v_link: tuple, p_net: PhysicalNetwork, solution: Solution) -> bool:
        """
        Undo the routing of a virtual link in the given solution.
        
        Args:
            v_link (tuple): The ID of the virtual node.
            p_net (PhysicalNetwork): The physical network object.
            solution (Solution): The solution object.

        Returns:
            bool: Return True if the route is successfully undone.
        """ 
        assert v_link in solution['link_paths'].keys()
        p_links = solution['link_paths'][v_link]
        for p_link in p_links:
            used_link_resources = solution['link_paths_info'][(v_link, p_link)]
            self.update_link_resources(p_net, p_link, used_link_resources, operator='+')
            del solution['link_paths_info'][(v_link, p_link)]
        del solution['link_paths'][v_link]
        return True

    def place_and_route(
            self,
            v_net: VirtualNetwork,
            p_net: PhysicalNetwork,
            v_node_id: int,
            p_node_id: int,
            solution: Solution,
            shortest_method: str = 'bfs_shortest',
            k: int = 1,
            check_feasibility: bool = True,
        ) -> Tuple[bool, dict]:
        """
        Attempt to place and route the virtual node `v_node_id` to the physical node `p_node_id` in the solution `solution`.        
        
        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            v_node_id (int): The virtual node ID.
            p_node_id (int): The physical node ID.
            solution (Solution): The solution.
            shortest_method (str): The shortest path method. Default: 'bfs_shortest'.
                                    ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
            k (int): The number of shortest paths to find. Default: 1.
            check_feasibility (bool): Whether to check the feasibility of the solution. Default: True.

        Returns:
            result (bool): The result of the placement and routing.
            check_info (dict): The check info of the placement and routing.
        """
        if check_feasibility:
            return self.safely_place_and_route(v_net, p_net, v_node_id, p_node_id, solution, shortest_method=shortest_method, k=k)
        else:
            return self.unsafely_place_and_route(v_net, p_net, v_node_id, p_node_id, solution, shortest_method=shortest_method, k=k)

    def safely_place_and_route(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_node_id: int, 
            p_node_id: int, 
            solution: Solution, 
            shortest_method: str = 'bfs_shortest', 
            k: int = 1
        ) -> Tuple[bool, dict]:
        """
        Attempt to place and route the virtual node `v_node_id` to the physical node `p_node_id` in the solution `solution`, ensuring all constraints are satisfied.      
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

    def unsafely_place_and_route(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_node_id: int, 
            p_node_id: int, 
            solution: Solution, 
            shortest_method: str = 'bfs_shortest', 
            k: int = 1
        ) -> Tuple[bool, dict]:
        """
        Attempt to place and route the virtual node `v_node_id` to the physical node `p_node_id` in the solution `solution`, without checking the feasibility of the solution.
        """
        assert shortest_method == 'k_shortest'
        # Place
        place_result, place_info = self.unsafely_place(v_net, p_net, v_node_id, p_node_id, solution)
        place_violation_value = solution['v_net_current_violation']
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
            route_result, route_info = self.unsafely_route(v_net, p_net, v_link, (p_node_id, n_p_node_id), solution, 
                                        shortest_method=shortest_method, k=k)
            route_violation_value = solution['v_net_current_violation']
            route_violation_value_list.append(route_violation_value)

        best_violation_value = place_violation_value + sum(route_violation_value_list)
        solution['v_net_current_violation'] = best_violation_value
        return True, route_info

    def undo_place_and_route(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, v_node_id: int, p_node_id:int, solution: Solution):
        """
        Undo the place and route operation, including the place and route of the neighbors of the virtual node.

        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            v_node_id (int): The ID of a virtual node.
            p_node_id (int): The ID of a physical node.
            solution (Solution): The solution class which contains the node slots and link paths.

        Returns:
            bool: True if the undo process is successful.
        """
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

    def node_mapping(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            sorted_v_nodes: list, 
            sorted_p_nodes: list, 
            solution: Solution, 
            reusable: bool = False, 
            inplace: bool = True, 
            matching_mathod: str = 'greedy',
            check_feasibility: bool = True
        ) -> bool:
        """
        Map all virtual nodes to physical nodes using the given matching method.

        Args:
            v_net (VirtualNetwork): Virtual network object.
            p_net (PhysicalNetwork): Physical network object.
            sorted_v_nodes (list): Sorted virtual nodes.
            sorted_p_nodes (list): Sorted physical nodes.
            solution (Solution): Solution object.
            reusable (bool, optional): Whether to reuse the existing mapping. Defaults to False.
            inplace (bool, optional): Whether to update the solution object. Defaults to True.
            matching_mathod (str, optional): Matching method. Defaults to 'greedy'.
                ['l2s2', 'greedy']
            check_feasibility (bool, optional): Whether to check feasibility. Defaults to True.

        Returns:
            bool: Whether the node mapping is successful.
        """
        if check_feasibility:
            return self.safely_node_mapping(v_net, p_net, sorted_v_nodes, sorted_p_nodes, solution, reusable, inplace, matching_mathod)
        else:
            return self.safely_node_mapping(v_net, p_net, sorted_v_nodes, sorted_p_nodes, solution, reusable, inplace, matching_mathod)
            # return self.unsafely_node_mapping(v_net, p_net, sorted_v_nodes, sorted_p_nodes, solution, reusable, inplace, matching_mathod)

    def safely_node_mapping(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            sorted_v_nodes: list, 
            sorted_p_nodes: list, 
            solution: Solution, 
            reusable: bool = False, 
            inplace: bool = True, 
            matching_mathod: str = 'greedy'
        ) -> bool:
        """
        Map all virtual nodes to physical nodes using the given matching method, ensuring all constraints are satisfied.
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

    def link_mapping(
            self,
            v_net: VirtualNetwork,
            p_net: PhysicalNetwork,
            solution: Solution,
            sorted_v_links: list = None,
            shortest_method: str = 'bfs_shortest',
            k: int = 10,
            inplace: bool = True,
            check_feasibility: bool = True
        ) -> bool:
        """
        Map all virtual links to physical paths using the given shorest path method.

        Args:
            v_net (object): Virtual network object.
            p_net (object): Physical network object.
            solution (dict): Solution dictionary.
            sorted_v_links (list, optional): List of sorted virtual links. Defaults to None.
            shortest_method (str, optional): Method used for shortest path calculation. Can be 'bfs_shortest', 'dijkstra_shortest', 'mcf'. Defaults to 'bfs_shortest'.
                ['bfs_shortest', 'dijkstra_shortest', 'mcf']
            k (int, optional): Number of paths to be calculated. Defaults to 10.
            inplace (bool, optional): Flag indicating whether the physical nodes should be modified in place. Defaults to True.

        Returns:
            bool: True if the mapping was successful, False otherwise.
        """
        if check_feasibility:
            return self.safely_link_mapping(v_net, p_net, solution, sorted_v_links, shortest_method, k, inplace)
        else:
            return self.unsafely_link_mapping(v_net, p_net, solution, sorted_v_links, shortest_method, k, inplace)

    def safely_link_mapping(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            solution: Solution, 
            sorted_v_links: list = None, 
            shortest_method: str = 'bfs_shortest', 
            k: int = 10, 
            inplace: bool = True
        ) -> bool:
        """
        Map all virtual links to physical paths using the given shorest path method, ensuring all constraints are satisfied.
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

    def unsafely_link_mapping(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            solution: Solution, 
            sorted_v_links: list = None, 
            shortest_method: str = 'bfs_shortest', 
            k: int = 10, 
            inplace: bool = True,
            pruning_ratio: float = None
        ) -> bool:
        """
        Map all virtual links to physical paths using the given shorest path method, without ensuring all constraints are satisfied.
        """
        p_net = p_net if inplace else copy.deepcopy(p_net)
        sorted_v_links = sorted_v_links if sorted_v_links is not None else list(v_net.links)
        node_slots = solution['node_slots']
        route_check_info_dict = {}
        violation_value_dict = {}
        sum_violation_value = 0
        for v_link_pair in sorted_v_links:
            p_path_pair = (node_slots[v_link_pair[0]], node_slots[v_link_pair[1]])
            route_result, route_check_info = self.unsafely_route(v_net, p_net, v_link_pair, p_path_pair, solution, shortest_method=shortest_method, k=k, pruning_ratio=pruning_ratio)
            violation_value = solution['v_net_current_violation']
            sum_violation_value += violation_value
            route_check_info_dict[v_link_pair] = route_check_info_dict
            violation_value_dict[v_link_pair] = violation_value
            if not route_result:
                # FAILURE
                solution.update({'route_result': False, 'result': False})
                return False, route_check_info_dict, 0
        # SUCCESS
        assert len(solution['link_paths']) == v_net.num_links
        solution['v_net_violation'] = sum_violation_value
        return True, route_check_info_dict

    def deploy(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution) -> bool:
        """
        Deploy a virtual network to a physical network with the given solution.
                
        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            solution (Solution): The solution of mapping virtual network to physical network.
        
        Returns:
            bool: True if deployment success, False otherwise.
        """
        if not solution['result']:
            return False
        for (v_node_id, p_node_id), used_node_resources in solution['node_slots_info'].items():
            self.update_node_resources(p_net, p_node_id, used_node_resources, operator='-')
        for (v_link, p_link), used_link_resources in solution['link_paths_info'].items():
            self.update_link_resources(p_net, p_link, used_link_resources, operator='-')
        return True

    def bfs_deploy(
            self, 
            v_net: 
            VirtualNetwork, 
            p_net: PhysicalNetwork, 
            sorted_v_nodes: list, 
            p_initial_node_id: int, 
            max_visit: int =100, 
            max_depth: int =10, 
            shortest_method: str = 'all_shortest', 
            k: int = 10
        ) -> Solution:
        """
        Deploy a virtual network to a physical network using BFS algorithm.

        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            sorted_v_nodes (list): The sorted virtual nodes.
            p_initial_node_id (int): The initial physical node id.
            max_visit (int, optional): The maximum number of visited nodes. Defaults to 100.
            max_depth (int, optional): The maximum depth of BFS. Defaults to 10.
            shortest_method (str, optional): The shortest path method. Defaults to 'all_shortest'.
                                                method: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
            k (int, optional): The number of shortest paths. Defaults to 10.

        Returns:
            Solution: The solution of mapping virtual network to physical network.
        """
        solution = Solution(v_net)

        max_visit_in_every_depth = int(np.power(max_visit, 1 / max_depth))
        
        curr_depth = 0
        visited = p_net.num_nodes * [False]
        queue = [(p_initial_node_id, curr_depth)]
        visited[p_initial_node_id] = True

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

    def deploy_with_node_slots(
            self,
            v_net: VirtualNetwork,
            p_net: PhysicalNetwork,
            node_slots: dict,
            solution: Solution,
            inplace: bool = True,
            shortest_method: str = 'bfs_shortest',
            k_shortest: int = 10,
            check_feasibility: bool = True
        ) -> bool:
        """
        Deploy a virtual network to a physical network with specified node slots.

        Args:
            v_net (VirtualNetwork): The virtual network to be deployed.
            p_net (PhysicalNetwork): The physical network to deploy on.
            node_slots (dict): A dictionary of node slots. Keys are the virtual nodes and values are the physical nodes.
            solution (Solution): The solution class which contains the node slots and link paths.
            inplace (bool, optional): Whether to operate on the original physical network or a copy. Defaults to True.
            shortest_method (str, optional): The method used to find the shortest path. Defaults to 'bfs_shortest'.
                ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
            k_shortest (int, optional): The number of shortest paths to find. Defaults to 10.
            check_feasibility (bool, optional): Whether to check the feasibility of the node slots. Defaults to True.
            
        Returns:
            bool: True if deployment success, False otherwise.
        """
        if check_feasibility:
            return self.safely_deploy_with_node_slots(v_net, p_net, node_slots, solution, inplace, shortest_method, k_shortest)
        else:
            return self.unsafely_deploy_with_node_slots(v_net, p_net, node_slots, solution, inplace, shortest_method, k_shortest)
            
    def safely_deploy_with_node_slots(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            node_slots: dict, 
            solution: Solution, 
            inplace: bool = True, 
            shortest_method: str = 'bfs_shortest', 
            k_shortest: int =10
        ) -> bool:
        """
        Deploy a virtual network to a physical network with specified node slots, ensuring all constraints are satisfied.
        """
        p_net = p_net if inplace else copy.deepcopy(p_net)
        # unfeasible solution
        if len(node_slots) != v_net.num_nodes or -1 in node_slots.values():
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
        # self.counter.count_solution(v_net, solution)
        return

    def unsafely_deploy_with_node_slots(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            node_slots: dict, 
            solution: Solution, 
            inplace: bool=True, 
            shortest_method='bfs_shortest', 
            k_shortest=10, 
            pruning_ratio=None
        ) -> bool:
        """
        Deploy a virtual network to a physical network with specified node slots, without checking the feasibility of the solution.
        """
        p_net = p_net if inplace else copy.deepcopy(p_net)
        # unfeasible solution
        if len(node_slots) != v_net.num_nodes or -1 in node_slots.values():
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
        link_mapping_result, route_check_info  = self.unsafely_link_mapping(v_net, p_net, 
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
        # self.counter.count_solution(v_net, solution)
        return

    def undo_deploy(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution) -> bool:
        """
        Undo the deployment of the virtual network in the physical network.

        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            solution (Solution): The solution class which contains the node slots and link paths.

        Returns:
            bool: True if the undo process is successful.
        """
        self.release(v_net, p_net, solution)
        solution.reset()
        return True

    def find_shortest_paths(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_link: tuple, 
            p_pair: tuple, 
            method: str = 'k_shortest', 
            k: int = 10,
            max_hop: float = 1e6
        ) -> list:
        """
        Find the shortest paths between two nodes in the physical network.

        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            v_link (tuple): The virtual link.
            p_pair (tuple): The physical pair.
            method (str, optional): The method to find the shortest paths. Defaults to 'k_shortest'.
                Optional methods: ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_shortest']
            k (int, optional): The number of shortest paths to find. Defaults to 10.
            max_hop (int, optional): The maximum number of hops. Defaults to 1e6.

        Returns:
            shortest_paths (list): The list of shortest paths.
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

    def create_available_network(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, v_link_pair):
        def available_link(n1, n2):
            p_link = p_net.links[(n1, n2)]
            result, info = self.check_link_constraints(v_net, p_net, v_link, p_link)
            return result
        v_link = v_net.links[v_link_pair]
        sub_graph = nx.subgraph_view(p_net, filter_edge=available_link)
        return sub_graph

    def create_pruned_network(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, v_link_pair, ratio=1., div=0.):
        """
        Create a pruned network from the original network.
        A virtual network embedding algorithm based on the connectivity of residual substrate network. In Proc. IEEE ICCSE, 2016.

        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            v_link_pair (tuple): The virtual link pair.
            ratio (float, optional): The ratio of the pruned network. Defaults to 1.
            div (float, optional): The div of the pruned network. Defaults to 0.

        Returns:
            Network: The pruned network.
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

    def route_v_links_with_mcf(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, v_link_list: list, solution: Solution):
        """
        Route virtual links with Minimum Cost Flow (MCF) algorithm.

        Args:
            v_net (VirtualNetwork): The virtual network.
            p_net (PhysicalNetwork): The physical network.
            v_link_list (list): The list of virtual links.
            solution (Solution): The solution class which contains the node slots and link paths.

        Returns:
            A tuple of (status, info). Status is True if the routing is successful, else False. Info is an empty dict.
        """
        from ortools.linear_solver import pywraplp
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
                    p_links = path_to_links(shortest_path)
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
        
    def find_bfs_shortest_path(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_link: tuple, 
            source: int, 
            target: int,
            weight: str = None
        ) -> list:
        """
        Find the shortest path from source to target in physical network.

        Args:
            v_net (VirtualNetwork): The virtual network graph.
            p_net (PhysicalNetwork): The physical network graph.
            v_link (list): List of virtual links.
            source (int): Source node id.
            target (int): Target node id.
            weight (str): Edge attribute name to use as weight. Defaults to None.

        Returns:
            list: A list of nodes in the shortest path from source to target. 
                If no path exists, return None.
        """
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
        
    def find_candidate_nodes(
            self, 
            v_net: VirtualNetwork, 
            p_net: PhysicalNetwork, 
            v_node_id: int, 
            filter: list = [], 
            check_node_constraint: bool = True,
            check_link_constraint: bool = True):
        """
        Find candidate nodes from physical network according to given virtual node. 
        
        Args:
            v_net (VirtualNetwork): The virtual network object.
            p_net (PhysicalNetwork): The physical network object.
            v_node_id (int): The virtual node id.
            filter (list, optional): The list of filtered nodes. Defaults to [].
            check_node_constraint (bool, optional): Whether to check node constraints. Defaults to True.
            check_link_constraint (bool, optional): Whether to check link constraints. Defaults to True.

        Returns:
            candidate_nodes (list): The list of candidate nodes.
        """
        all_p_nodes = np.array(list(p_net.nodes))
        if check_node_constraint:
            suitable_nodes = [p_node_id for p_node_id in all_p_nodes if self.check_node_constraints(v_net, p_net, v_node_id, p_node_id)[0]]
            candidate_nodes = list(set(suitable_nodes).difference(set(filter)))
        else:
            candidate_nodes = []
        if check_link_constraint:
            aggr_method = 'sum' if self.shortest_method == 'mcf' else 'max'
            # checked_nodes = candidate_nodes_with_node_constraint if check_node_constraint else list(p_net.nodes)
            v_node_degrees = np.array(list(dict(v_net.degree()).values()))
            p_node_degrees = np.array(list(dict(p_net.degree()).values()))
            v_link_aggr_resource = np.array(v_net.get_aggregation_attrs_data(self.link_resource_attrs, aggr=aggr_method))
            p_link_aggr_resource = np.array(p_net.get_aggregation_attrs_data(self.link_resource_attrs, aggr=aggr_method))
            degrees_comparison = p_node_degrees[:] >= v_node_degrees[v_node_id]
            resource_comparison = np.all(v_link_aggr_resource[:, [v_node_id]] <= p_link_aggr_resource[:, :], axis=0)
            suitable_nodes = all_p_nodes[np.logical_and(degrees_comparison, resource_comparison)]
            new_filter = set(all_p_nodes) - set(candidate_nodes)
            candidate_nodes = list(set(candidate_nodes).difference(new_filter))
        else:
            candidate_nodes = candidate_nodes
        return candidate_nodes

    def find_feasible_nodes(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, v_node_id, node_slots):
        """
        Find feasible nodes in physical network for a given virtual node.

        Args:
            v_net (VirtualNetwork): Virtual Network
            p_net (PhysicalNetwork): Physical Network
            v_node_id (int): ID of the virtual node
            node_slots (dict): Dictionary of node slots, where key is virtual node id and value is physical node id

        Returns:
            feasible_nodes (list): List of feasible physical nodes ids
        """
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

    def construct_candidates_dict(self, v_net: VirtualNetwork, p_net: PhysicalNetwork):
        """
        Constructs a dictionary of candidates for each node in v_net.
        
        Args:
            v_net (VirtualNetwork): Virtual network object.
            p_net (PhysicalNetwork): Physical network object.

        Returns:
            candidates_dict (dict): A dictionary mapping each node in v_net to its candidate nodes in p_net.
        """
        candidates_dict = {}
        for v_node_id in list(v_net.nodes):
            candidate_nodes = self.find_candidate_nodes(v_net, p_net, v_node_id)
            candidates_dict[v_node_id] = candidate_nodes
        return candidates_dict

    def release(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution):
        """
        Release resources occupied by the Virtual network of the physical network, when the virtual network leaves the physical network.
        
        Args:
            v_net (VirtualNetwork): The virtual network object.
            p_net (PhysicalNetwork): The physical network object.
            solution (Solution): The mapping solution.

        Returns:
            bool: True if the release is successful, False otherwise.
        """
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


if __name__ == '__main__':
    pass
