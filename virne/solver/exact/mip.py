# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

import pprint

from virne.core import Solution
from virne.core.environment import SolutionStepEnvironment
from virne.solver.base_solver import Solver, SolverRegistry


@SolverRegistry.register(solver_name='mip', solver_type='exact')
class MipSolver(Solver):
    """
    An exact solver based on Mixed Integer Programming (MIP) with OR-Tools.

    References:
        - Mosharaf Chowdhury et al. "ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping". In TON, 2012.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(MipSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'mcf')
        self.k_shortest = kwargs.get('k_shortest', 10)
        self.META_BW = 9999
        self.MAX_TIME_IN_SECONDS = 10
        self.OR_SOLVER = 'mip'
        if self.OR_SOLVER == 'cp':
            self.solver_with_or_tools = self.solve_with_cp
        else:
            self.solver_with_or_tools = self.solve_with_mip


    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']
        self.solution = Solution.from_v_net(v_net)
        self.solver_with_or_tools(v_net, p_net)
        return self.solution


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


    def solve_with_cp(self, v_net, p_net):

        num_p_nodes = p_net.number_of_nodes()
        num_v_nodes = v_net.number_of_nodes()

        p_node_list = list(range(num_p_nodes))
        v_node_list = list(range(num_v_nodes))
        m_node_list = list(range(num_p_nodes, num_p_nodes + num_v_nodes))
        a_node_list = list(range(num_p_nodes + num_v_nodes))


        n_attr_name_list = ['cpu']
        e_attr_name_list = ['bw']
        candidates_dict = self.controller.construct_candidates_dict(v_net, p_net)
        # {m: [p for p in p_node_list] for m in m_node_list}
        node_resource_dict, edge_resource_dict = self.construct_resource_dict(v_net, p_net, n_attr_name_list, e_attr_name_list, candidates_dict)
        assert len(e_attr_name_list) == 1

        model = cp_model.CpModel()

        # x varibles
        x = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                x[(n_id_a, n_id_b)] = model.NewBoolVar(f'x({n_id_a, n_id_b})')
                # x[(n_id_a, n_id_b)] = model.NewIntervalVar(f'x({n_id_a, n_id_b})')

        # f varibles
        f = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                for v_edge in v_net.links:
                    f[(n_id_a, n_id_b, v_edge)] = model.NewIntVar(lb=0, ub=self.META_BW, name=f'f({n_id_a, n_id_b, v_edge})')

        # Objective
        model.Minimize(
            sum(f[(u, v, i)] for i in list(v_net.links) for u in list(p_net.nodes) for v in list(p_net.nodes))
            + \
            sum(x[m, w] * node_resource_dict[m][n_attr_name] for m in m_node_list for w in p_node_list for n_attr_name in n_attr_name_list)
        )

        # Capacity constraint
        for n_attr_name in n_attr_name_list:
            for m in m_node_list:
                for w in p_node_list:
                    if node_resource_dict[m][n_attr_name] == 0:
                        break
                    model.Add(x[(m,w)] * node_resource_dict[m][n_attr_name] <= node_resource_dict[w][n_attr_name])

        for e_attr_name in e_attr_name_list:
            for u in a_node_list:
                for v in a_node_list:
                    sum_flow = sum([f[(u, v, i)] + f[(v, u, i)] for i in v_net.links])
                    model.Add(sum_flow <= edge_resource_dict[(u,v)][e_attr_name] * x[(u, v)])

        for e_attr_name in e_attr_name_list:
            for i in v_net.links:
                for n_id in a_node_list:
                    if n_id == i[0] + num_p_nodes:
                        model.Add(
                            sum(f[n_id, w, i] for w in a_node_list) - \
                            sum(f[w, n_id, i] for w in a_node_list) == 1 * v_net.links[i][e_attr_name]
                        )
                    elif n_id == i[1] + num_p_nodes:
                        model.Add(
                            sum(f[n_id, w, i] for w in a_node_list) - \
                            sum(f[w, n_id, i] for w in a_node_list) == -1 * v_net.links[i][e_attr_name]
                        )
                    else:
                        model.Add(
                            sum(f[(n_id, w, i)] for w in a_node_list) - sum(f[(w, n_id, i)] for w in a_node_list) == 0
                        )

        # Meta constraint
        for m in m_node_list:
            model.Add(sum(x[(m,w)] for w in candidates_dict[m-num_p_nodes]) == 1)

        for w in p_node_list:
            model.Add(sum(x[(m,w)] for m in m_node_list) <= 1)

        for e_attr_name in e_attr_name_list:
            for u in a_node_list:
                for v in a_node_list:
                    model.Add(x[(u,v)] <= edge_resource_dict[(u,v)][e_attr_name])

        for u in a_node_list:
            for v in a_node_list:
                model.Add(x[(u,v)] == x[(v,u)])

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.MAX_TIME_IN_SECONDS
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # print(f'Maximum of objective function: {solver.ObjectiveValue()}\n')
            node_slots = {}
            node_slots_info = {}
            link_paths = {}
            link_paths_info = {}
            for u in a_node_list:
                for v in a_node_list:
                    if solver.Value(x[(u, v)]):
                        if (u, v) not in p_net.links and u in p_net.nodes:
                            node_slots[v-num_p_nodes] = u
                            node_slots_info[(v-num_p_nodes, u)] = {n_attr_name: v_net.nodes[v-num_p_nodes][n_attr_name] for n_attr_name in n_attr_name_list}
                        
            for i in v_net.links:
                link_paths[i] = []
                for u in a_node_list:
                    for v in a_node_list:
                            if solver.Value(f[(u, v, i)]) and u in p_net.nodes and v in p_net.nodes:
                                link_paths[i].append((u, v))
                                link_paths_info[(i, (u, v))] = {e_attr_name_list[0]: solver.Value(f[(u, v, i)])}
                                # print(f[(u, v, i)], solver.Value(f[(u, v, i)]))
            # print(sum(solver.Value(x[(i, j)]) for i in a_node_list for j in a_node_list))
            pprint.pprint(node_slots_info)
            pprint.pprint(link_paths_info)

            self.solution['result'] = True
            self.solution['node_slots'] = node_slots
            self.solution['link_paths'] = link_paths
            self.solution['node_slots_info'] = node_slots_info
            self.solution['link_paths_info'] = link_paths_info


    def solve_with_mip(self, v_net, p_net):

        num_p_nodes = p_net.number_of_nodes()
        num_v_nodes = v_net.number_of_nodes()

        p_node_list = list(range(num_p_nodes))
        v_node_list = list(range(num_v_nodes))
        m_node_list = list(range(num_p_nodes, num_p_nodes + num_v_nodes))
        a_node_list = list(range(num_p_nodes + num_v_nodes))


        n_attr_name_list = ['cpu']
        e_attr_name_list = ['bw']
        candidates_dict = self.controller.construct_candidates_dict(v_net, p_net)
        # {m: [p for p in p_node_list] for m in m_node_list}
        node_resource_dict, edge_resource_dict = self.construct_resource_dict(v_net, p_net, n_attr_name_list, e_attr_name_list, candidates_dict)
        assert len(e_attr_name_list) == 1

        solver = pywraplp.Solver.CreateSolver('SCIP')

        # x varibles
        x = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                x[(n_id_a, n_id_b)] = solver.IntVar(lb=0, ub=1, name=f'x({n_id_a, n_id_b})')

        # f varibles
        f = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                for v_edge in v_net.links:
                    f[(n_id_a, n_id_b, v_edge)] = solver.IntVar(lb=0, ub=self.META_BW, name=f'f({n_id_a, n_id_b, v_edge})')

        # Objective
        solver.Minimize(
            sum(f[(u, v, i)] for i in list(v_net.links) for u in list(p_net.nodes) for v in list(p_net.nodes))
            + \
            sum(x[m, w] * node_resource_dict[m][n_attr_name] for m in m_node_list for w in p_node_list for n_attr_name in n_attr_name_list)
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
            node_slots = {}
            node_slots_info = {}
            link_paths = {}
            link_paths_info = {}
            for u in a_node_list:
                for v in a_node_list:
                    if x[(u, v)].solution_value():
                        if (u, v) not in p_net.links and u in p_net.nodes:
                            node_slots[v-num_p_nodes] = u
                            node_slots_info[(v-num_p_nodes, u)] = {n_attr_name: v_net.nodes[v-num_p_nodes][n_attr_name] for n_attr_name in n_attr_name_list}
                        
            for i in v_net.links:
                link_paths[i] = []
                for u in a_node_list:
                    for v in a_node_list:
                            if f[(u, v, i)].solution_value() and u in p_net.nodes and v in p_net.nodes:
                                link_paths[i].append((u, v))
                                link_paths_info[(i, (u, v))] = {e_attr_name_list[0]: f[(u, v, i)].solution_value()}
            pprint.pprint(node_slots_info)
            pprint.pprint(link_paths_info)

            self.solution['result'] = True
            self.solution['node_slots'] = node_slots
            self.solution['link_paths'] = link_paths
            self.solution['node_slots_info'] = node_slots_info
            self.solution['link_paths_info'] = link_paths_info
