# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import random

from virne.core import Solution
from virne.core.environment import SolutionStepEnvironment
from virne.solver.base_solver import Solver, SolverRegistry


@SolverRegistry.register(solver_name='r_round', solver_type='rounding')
class RandomizedRoundingSolver(Solver):
    """
    An approximation solver based on randomized rounding algorithm.
    
    References:
        - Mosharaf Chowdhury et al. "ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping". In TON, 2012.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(RandomizedRoundingSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'bfs_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)
        self.META_BW = 9999
        self.MAX_TIME_IN_SECONDS = 10
        self.solver_with_or_tools = self.solve_with_mip

    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']
        candidates_dict = self.controller.construct_candidates_dict(v_net, p_net)
        v_p_value_dict = self.solver_with_or_tools(v_net, p_net, candidates_dict)
        if v_p_value_dict is not None:
            solution = self.deploy_with_v_p_value_dict(v_net, p_net, v_p_value_dict, candidates_dict)
        else:
            solution = Solution.from_v_net(v_net)
        return solution

    def deploy_with_v_p_value_dict(self, v_net, p_net, v_p_value_dict, candidates_dict):
        solution = Solution.from_v_net(v_net)
        for v_node_id in list(v_net.nodes):
            selected_p_net_nodes = list(solution['node_slots'].values())
            v_p_candidate_prob_dict = {p_node_id: v_p_value_dict[v_node_id][p_node_id] for p_node_id in list(p_net.nodes)
                            if p_node_id in candidates_dict[v_node_id] and p_node_id not in selected_p_net_nodes}
            if len(v_p_candidate_prob_dict) == 0:
                # Failure
                solution['place_result'] = False
                return solution
            p_node_id = self.select_p_net_node(v_p_candidate_prob_dict)
            place_and_route_result, place_and_route_info = self.controller.place_and_route(v_net, p_net, v_node_id, p_node_id, solution, 
                                                shortest_method=self.shortest_method, k=self.k_shortest)
            if not place_and_route_result:
                # Failure
                solution['route_result'] = False
                return solution
        # Success
        solution['result'] = True
        return solution

    def select_p_net_node(self, v_p_candidate_prob_dict):
        if sum(list(v_p_candidate_prob_dict.values())) == 0:
            return random.choices(list(v_p_candidate_prob_dict.keys()), k=1)[0]
        selected_p_node_id = random.choices(list(v_p_candidate_prob_dict.keys()), weights=list(v_p_candidate_prob_dict.values()), k=1)[0]
        return selected_p_node_id

    def construct_resource_dict(self, v_net, p_net, n_attr_name_list, e_attr_name_list, candidates_dict):

        def get_node_type(n_id):
            return 'p' if n_id < num_p_nodes else 'v'

        def get_node_resource_dict(n_id, n_attr_name):
            n_type = get_node_type(n_id)
            if n_type == 'p':
                return p_net.nodes[n_id][n_attr_name]
            elif n_type == 'v':
                return v_net.nodes[n_id - num_p_nodes][n_attr_name]

        def get_edge_resource_dict(e_id, e_attr_name):
            u, v = e_id
            u_type = get_node_type(u)
            v_type = get_node_type(v)
            # p-p
            if u_type == 'p' and v_type == 'p':
                if (u, v) in p_net.links:
                    return p_net.links[(u, v)][e_attr_name]
                return 0
            elif u_type == 'v' and v_type == 'v':
                return 0
            else:
                if u > v:
                    candidates = candidates_dict[u-num_p_nodes]
                    return self.META_BW if v in candidates else 0
                else:
                    candidates = candidates_dict[v-num_p_nodes]
                    return self.META_BW if u in candidates else 0
        
        num_p_nodes = p_net.number_of_nodes()
        num_v_nodes = v_net.number_of_nodes()

        node_resource_dict = {}
        for n_id in range(num_p_nodes+num_v_nodes):
            node_resource_dict[n_id] = {}
            for n_attr_name in n_attr_name_list:
                node_resource_dict[n_id][n_attr_name] = get_node_resource_dict(n_id, n_attr_name)

        edge_resource_dict = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                e_id = (n_id_a, n_id_b)
                edge_resource_dict[e_id] = {}
                for e_attr_name in e_attr_name_list:
                    edge_resource_dict[e_id][e_attr_name] = get_edge_resource_dict(e_id, e_attr_name)

        return node_resource_dict, edge_resource_dict


    def solve_with_mip(self, v_net, p_net, candidates_dict):
        from ortools.linear_solver import pywraplp
        num_p_nodes = p_net.number_of_nodes()
        num_v_nodes = v_net.number_of_nodes()

        p_node_list = list(range(num_p_nodes))
        v_node_list = list(range(num_v_nodes))
        m_node_list = list(range(num_p_nodes, num_p_nodes + num_v_nodes))
        a_node_list = list(range(num_p_nodes + num_v_nodes))


        n_attr_name_list = ['cpu']
        e_attr_name_list = ['bw']
        # {m: [p for p in p_node_list] for m in m_node_list}
        node_resource_dict, edge_resource_dict = self.construct_resource_dict(v_net, p_net, n_attr_name_list, e_attr_name_list, candidates_dict)
        assert len(e_attr_name_list) == 1

        solver = pywraplp.Solver.CreateSolver('Glop')

        # x varibles
        x = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                x[(n_id_a, n_id_b)] = solver.NumVar(lb=0, ub=1, name=f'x({n_id_a, n_id_b})')

        # f varibles
        f = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                for v_edge in v_net.links:
                    f[(n_id_a, n_id_b, v_edge)] = solver.NumVar(lb=0, ub=self.META_BW, name=f'f({n_id_a, n_id_b, v_edge})')

        # Objective
        # solver.Minimize(
        #     sum(f[(u, v, i)] for i in list(v_net.links) for u in list(p_net.nodes) for v in list(p_net.nodes))
        #     + \
        #     sum(x[m, w] * node_resource_dict[m][n_attr_name] for m in m_node_list for w in p_node_list for n_attr_name in n_attr_name_list)
        # )

        solver.Minimize(
            sum(1 / (edge_resource_dict[(u,v)][e_attr_name] + 1e-6) * sum(f[(u, v, i)] for i in list(v_net.links)) for u in list(p_net.nodes) for v in list(p_net.nodes) for e_attr_name in e_attr_name_list)
            + \
            sum(1 / (node_resource_dict[w][n_attr_name] + 1e-6) * sum(x[m, w] * node_resource_dict[m][n_attr_name] for m in m_node_list) for w in p_node_list for n_attr_name in n_attr_name_list)
        )

        # Capacity constraint
        for n_attr_name in n_attr_name_list:
            for m in m_node_list:
                for w in p_node_list:
                    if node_resource_dict[m][n_attr_name] == 0:
                        break
                    solver.Add(x[(m,w)] * node_resource_dict[m][n_attr_name] <= node_resource_dict[w][n_attr_name])

        for e_attr_name in e_attr_name_list:
            for u in a_node_list:
                for v in a_node_list:
                    sum_flow = sum([f[(u, v, i)] + f[(v, u, i)] for i in v_net.links])
                    solver.Add(sum_flow <= edge_resource_dict[(u,v)][e_attr_name] * x[(u, v)])

        for e_attr_name in e_attr_name_list:
            for i in v_net.links:
                for n_id in a_node_list:
                    if n_id == i[0] + num_p_nodes:
                        solver.Add(
                            sum(f[n_id, w, i] for w in a_node_list) - \
                            sum(f[w, n_id, i] for w in a_node_list) == 1 * v_net.links[i][e_attr_name]
                        )
                    elif n_id == i[1] + num_p_nodes:
                        solver.Add(
                            sum(f[n_id, w, i] for w in a_node_list) - \
                            sum(f[w, n_id, i] for w in a_node_list) == -1 * v_net.links[i][e_attr_name]
                        )
                    else:
                        solver.Add(
                            sum(f[(n_id, w, i)] for w in a_node_list) - sum(f[(w, n_id, i)] for w in a_node_list) == 0
                        )

        # Meta constraint
        for m in m_node_list:
            solver.Add(sum(x[(m,w)] for w in candidates_dict[m-num_p_nodes]) == 1)

        for w in p_node_list:
            solver.Add(sum(x[(m,w)] for m in m_node_list) <= 1)

        for e_attr_name in e_attr_name_list:
            for u in a_node_list:
                for v in a_node_list:
                    solver.Add(x[(u,v)] <= edge_resource_dict[(u,v)][e_attr_name])

        for u in a_node_list:
            for v in a_node_list:
                solver.Add(x[(u,v)] == x[(v,u)])

        solver.SetTimeLimit(self.MAX_TIME_IN_SECONDS * 1000)
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE :
            # print('Solution:')
            # print('Objective value =', solver.Objective().Value())
            v_p_value_dict = {}
            for m in m_node_list:
                v_p_value_dict[m-num_p_nodes] = []
                for p in p_node_list:
                    if p not in candidates_dict[m-num_p_nodes]:
                        v_p_value_dict[m-num_p_nodes].append(0.)
                    else:
                        v_p_value_dict[m-num_p_nodes].append(x[(m,p)].solution_value() * (sum(f[(m, p, i)].solution_value() for i in v_net.links)) + sum(f[(p, m, i)].solution_value() for i in v_net.links))
            # pprint.pprint(v_p_value_dict)
            return v_p_value_dict
        else:
            return None
