import copy
import random
import pulp
import pandas as pd
from collections import defaultdict

from base import Controller, Solution
from ..solver import Solver
from ..rank.node_rank import OrderNodeRank, GRCNodeRank, FFDNodeRank, NRMNodeRank, RWNodeRank, RandomNodeRank
from ..rank.edge_rank import OrderEdgeRank, FFDEdgeRank


class DeterministicRoundingSolver(Solver):
    
    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        super(DeterministicRoundingSolver, self).__init__(name=name, reusable=reusable, verbose=verbose, **kwargs)
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'bfs_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)

    def solve(self, instance):
        vn, pn  = instance['vn'], instance['pn']

        self.solution = Solution(vn)
        node_mapping_result = self.node_mapping(vn, pn)
        if node_mapping_result:
            link_mapping_result = self.link_mapping(vn, pn)
            if link_mapping_result:
                # SUCCESS
                self.solution['result'] = True
                return self.solution
            else:
                # FAILURE
                self.solution['route_result'] = False
        else:
            # FAILURE
            self.solution['place_result'] = False
        self.solution['result'] = False
        return self.solution

    def node_mapping(self, vn, pn):
        r"""Attempt to accommodate VNF in appropriate physical node."""
        vn_rank = self.node_rank(vn)
        pn_rank = self.node_rank(pn)
        sorted_v_nodes = list(vn_rank)
        sorted_p_nodes = list(pn_rank)
        
        node_mapping_result = Controller.node_mapping(vn, pn, 
                                                        sorted_v_nodes=sorted_v_nodes, 
                                                        sorted_p_nodes=sorted_p_nodes, 
                                                        solution=self.solution, 
                                                        reusable=False, inplace=True, matching_mathod=self.matching_mathod)
        return node_mapping_result

    def link_mapping(self, vn, pn):
        r"""Seek a path connecting """
        if self.edge_rank is None:
            sorted_v_edges = vn.edges
        else:
            vn_edges_rank_dict = self.edge_rank(vn)
            vn_edges_sort = sorted(vn_edges_rank_dict.items(), reverse=True, key=lambda x: x[1])
            sorted_v_edges = [edge_value[0] for edge_value in vn_edges_sort]

        link_mapping_result = Controller.link_mapping(vn, pn, solution=self.solution, 
                                                        sorted_v_edges=sorted_v_edges, 
                                                        shortest_method=self.shortest_method,
                                                        k=self.k_shortest, inplace=True)
        return link_mapping_result

    def construct_augmented_network(vn, pn):
        augmented_network = copy.deepcopy(pn)
        for v_node_id, v_node_data in vn.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']
            # Meta node add
            meta_node_id = v_node_id + pn.num_nodes
            augmented_network.add_node(meta_node_id)
            for key, value in v_node_data.kwargs():
                augmented_network.nodes[meta_node_id][key] = value

            # Meta edge add
            for a_node_id, a_node_data, in augmented_network.nodes(data=True):
                if a_node_id >= pn.num_nodes:
                    continue
                # TODO: according to node location constraints
                if random.random() > 0.8:
                    augmented_network.add_edge(meta_node_id, a_node_id)
                    for e_attr in vn.get_edge_attrs(['resource']):
                        augmented_network.edges[meta_node_id, a_node_id].update({e_attr.name: 1000000})
        return augmented_network

    def calc_LP_variables(self, augmented_network, vn, pn):
        num_nodes = len(list(augmented_network.net.nodes))
        edges_bandwidth = [[0] * num_nodes for _ in range(num_nodes)]
        a_nodes_id = []
        s_nodes_id = []
        meta_nodes_id = []
        nodes_CPU = []
        v_flow_id = []
        v_flow_start = []
        v_flow_end = []
        v_flow_demand = []
        # location_ids = defaultdict(list)
        # meta_nodes_location = {}

        for a_edge_src, a_edge_dst, a_edge_data in augmented_network.net.edges(data=True):
            edges_bandwidth[a_edge_src][a_edge_dst] = a_edge_data['bw']
            edges_bandwidth[a_edge_dst][a_edge_src] = a_edge_data['bw']

        for a_node_id, a_node_data in augmented_network.net.nodes(data=True):
            a_nodes_id.append(a_node_id)
            nodes_CPU.append(a_node_data['cpu'])
            if a_node_id >= pn.num_nodes:
                meta_nodes_id.append(a_node_id)
                # meta_nodes_location[a_node_id] = a_node_data['LOCATION']
            else:
                s_nodes_id.append(a_node_id)
                # location_ids[a_node_data['LOCATION']].append(a_node_id)

        id_idx = 0
        for v_edge_src, v_edge_dst, v_edge_data in vn.edges(data=True):
            v_flow_id.append(id_idx)
            v_flow_start.append(v_edge_src + pn.num_nodes)
            v_flow_end.append(v_edge_dst + pn.num_nodes)
            v_flow_demand.append(v_edge_data['bw'])
            id_idx += 1

        # f_vars
        f_vars = {
            (i, u, v): pulp.LpVariable(
                cat=pulp.LpContinuous,
                lowBound=0,
                name="f_{0}_{1}_{2}".format(i, u, v)
            )
            for i in v_flow_id for u in a_nodes_id for v in a_nodes_id
        }

        # x_vars
        x_vars = {(u, v):
            pulp.LpVariable(
                cat=pulp.LpContinuous,
                lowBound=0, upBound=1,
                name="x_{0}_{1}".format(u, v)
            )
            for u in a_nodes_id for v in a_nodes_id
        }

        opt_model = pulp.LpProblem(name="MIP Model", sense=pulp.LpMinimize)
        # Objective function
        opt_model += sum(1 / (edges_bandwidth[u][v] + 0.000001) *
                         sum(f_vars[i, u, v] for i in v_flow_id)
                         for u in s_nodes_id for v in s_nodes_id) + \
                     sum(1 / (nodes_CPU[w] + 0.000001) *
                         sum(x_vars[m, w] * nodes_CPU[m]
                             for m in meta_nodes_id) for w in s_nodes_id)

        # Capacity constraint 1
        for u in a_nodes_id:
            for v in a_nodes_id:
                opt_model += sum(f_vars[i, u, v] + f_vars[i, v, u] for i in v_flow_id) <= edges_bandwidth[u][v]

        # Capacity constraint 2
        for m in meta_nodes_id:
            for w in s_nodes_id:
                opt_model += nodes_CPU[w] >= x_vars[m, w] * nodes_CPU[m]

        # Flow constraints 1
        for i in v_flow_id:
            for u in s_nodes_id:
                opt_model += sum(f_vars[i, u, w] for w in a_nodes_id) - \
                             sum(f_vars[i, w, u] for w in a_nodes_id) == 0

        # Flow constraints 2
        for i in v_flow_id:
            for fs in v_flow_start:
                opt_model += sum(f_vars[i, fs, w] for w in a_nodes_id) - \
                             sum(f_vars[i, w, fs] for w in a_nodes_id) == v_flow_demand[i]

        # Flow constraints 3
        for i in v_flow_id:
            for fe in v_flow_end:
                opt_model += sum(f_vars[i, fe, w] for w in a_nodes_id) - \
                             sum(f_vars[i, w, fe] for w in a_nodes_id) == -1 * v_flow_demand[i]

        # Meta constraint 1
        for w in s_nodes_id:
            opt_model += sum(x_vars[m, w] for m in meta_nodes_id) <= 1

        # Meta constraint 2
        for u in a_nodes_id:
            for v in a_nodes_id:
                opt_model += x_vars[u, v] == x_vars[v, u]

        # # Meta constraint 3
        # for m in meta_nodes_id:
        #     opt_model += sum(x_vars[m, w] for w in location_ids[meta_nodes_location[m]]) == 1

        # for minimization
        # solve VNE_LP_RELAX
        opt_model.solve(pulp.PULP_CBC_CMD(msg=0))

        # for v in opt_model.variables():
        #     if v.varValue > 0:
        #         print(v.name, "=", v.varValue)

        # make the DataFrame for f_vars and x_vars
        opt_lp_f_vars = pd.DataFrame.from_dict(f_vars, orient="index", columns=['variable_object'])
        opt_lp_f_vars.index = pd.MultiIndex.from_tuples(opt_lp_f_vars.index, names=["i", "u", "v"])
        opt_lp_f_vars.reset_index(inplace=True)
        opt_lp_f_vars["solution_value"] = opt_lp_f_vars["variable_object"].apply(lambda item: item.varValue)
        opt_lp_f_vars.drop(columns=["variable_object"], inplace=True)

        opt_lp_x_vars = pd.DataFrame.from_dict(x_vars, orient="index", columns=['variable_object'])
        opt_lp_x_vars.index = pd.MultiIndex.from_tuples(opt_lp_x_vars.index, names=["u", "v"])
        opt_lp_x_vars.reset_index(inplace=True)
        opt_lp_x_vars["solution_value"] = opt_lp_x_vars["variable_object"].apply(lambda item: item.varValue)
        opt_lp_x_vars.drop(columns=["variable_object"], inplace=True)

        return opt_lp_f_vars, opt_lp_x_vars