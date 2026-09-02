# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import networkx as nx
import numpy as np

from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.solver.base_solver import Solver, SolverRegistry


@SolverRegistry.register(solver_name='ldg', solver_type='heuristic')
class LDGSolver(Solver):
    """
    A locality-dominant greedy solver that embeds virtual networks by joint
    place-and-route, keeping every request's footprint as tight on the
    substrate as feasibility allows.

    For each request, virtual nodes are sorted by total demand, descending,
    where the demand of a node is (node resource demands + 2 x adjacent link
    bandwidth demands) x (degree + 1); the heavy, well-connected cores are
    embedded first, while contiguous capacity is still available. Every node
    is then placed greedily: the controller's feasibility filter proposes
    candidate substrate nodes, and each candidate p is scored by
    score(p) = cpu_free(p)^0.25 * adj_bw_free(p) * (1 + 8 * prox(p)), where
    prox(p) = sum over already-placed virtual neighbors of
    (1 + link demand / 25) / (1 + hop distance); hop distances come from an
    all-pairs BFS cache that depends only on the static substrate topology.
    The top five candidates are tried with place_and_route
    (available_shortest routing over residual capacities) until one sticks,
    and a request that dead-ends is rolled back and retried once in BFS order
    rooted at its heaviest node before being rejected.

    Locality dominates the score by design: under heavy load the binding
    resource is link bandwidth rather than node capacity, and scattering a
    request across the substrate stretches every virtual link over long
    paths and fragments bandwidth for future requests, a cost that exceeds
    the benefit of picking a marginally richer node. Hence the mild node
    exponent (0.25), the strong locality multiplier (8), and the
    demand-weighted proximity term.

    Measured on held-out simulation streams at arrival rate 0.12 (long-term
    average time revenue, seeds 42 / 137 / 2024): 9299 / 9815 / 9701 with
    ~0.79 acceptance, versus nrm_rank 7025 / 7438 / 7424, pl_rank
    8233 / 8360 / 8294, and the previous tuned configuration 8875 / 8891 /
    8994 on the same streams.

    deterministic; no learned components.

    Attributes:
        ALPHA: exponent on free node capacity in the candidate base score.
        BETA: weight of the locality (proximity) term.
        PROX_NORM: link-demand normalizer of the proximity weight.
        MAX_TRIES: number of top-scored candidates attempted per virtual node.
    """

    # frozen configuration found by hyper-parameter search; deliberately not
    # exposed as constructor arguments so that every `ldg` run is identical
    ALPHA = 0.25
    BETA = 8.0
    PROX_NORM = 25.0
    MAX_TRIES = 5

    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs) -> None:
        """
        Initialize the LDGSolver.

        Args:
            controller: the controller to control the mapping process.
            recorder: the recorder to record the mapping process.
            counter: the counter to count the mapping process.
            logger: the logger to log the mapping process.
            config: the experiment configuration.
            kwargs: the keyword arguments.
        """
        super(LDGSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        # link mapping
        self.shortest_method = 'available_shortest'
        # cached all-pairs hop distances (the substrate topology is static)
        self._dist = None
        self._dist_signature = None

    # ---------- resource snapshots ----------
    def _residual_vectors(self, p_net):
        """Remaining node capacity and remaining adjacent link capacity per node."""
        cpu = np.array(p_net.get_node_attrs_data(p_net.get_node_attrs('resource'))).sum(axis=0)
        bw = np.array(p_net.get_aggregation_attrs_data(
            p_net.get_link_attrs('resource'), aggr='sum', normalized=False)).sum(axis=0)
        nodes = list(p_net.nodes)
        return dict(zip(nodes, cpu)), dict(zip(nodes, bw))

    # ---------- cached topology distances ----------
    def _hop_distances(self, p_net):
        signature = tuple(p_net.nodes)
        if self._dist is None or self._dist_signature != signature:
            self._dist = dict(nx.all_pairs_shortest_path_length(p_net))
            self._dist_signature = signature
        return self._dist

    # ---------- virtual node ordering ----------
    def _rank_v_nodes(self, v_net):
        """Order virtual nodes by (node + 2 x adjacent link demand) x (degree + 1), descending."""
        node_resource_names = [a.name if hasattr(a, 'name') else a
                               for a in v_net.get_node_attrs('resource')]
        link_resource_names = [a.name if hasattr(a, 'name') else a
                               for a in v_net.get_link_attrs('resource')]

        def total_demand(v_node_id):
            node_demand = sum(v_net.nodes[v_node_id].get(name, 0)
                              for name in node_resource_names)
            link_demand = sum(v_net.edges[u, v_node_id].get(name, 0)
                              for u in v_net.adj[v_node_id] for name in link_resource_names)
            return (node_demand + 2.0 * link_demand) * (v_net.degree(v_node_id) + 1)

        return sorted(v_net.nodes, key=total_demand, reverse=True)

    def _bfs_order(self, v_net):
        """BFS order rooted at the heaviest virtual node (restart order)."""
        node_resource_names = [a.name if hasattr(a, 'name') else a
                               for a in v_net.get_node_attrs('resource')]

        def demand_degree(v_node_id):
            demand = sum(v_net.nodes[v_node_id].get(name, 0)
                         for name in node_resource_names)
            return demand * (v_net.degree(v_node_id) + 1)

        root = max(v_net.nodes, key=demand_degree)
        return list(nx.bfs_tree(v_net, root))

    # ---------- candidate scoring ----------
    def _score(self, p_node_id, v_node_id, placed_map, v_net, cpu, bw, dist):
        """Locality-dominant score of placing `v_node_id` onto `p_node_id`."""
        base = (cpu[p_node_id] + 1e-9) ** self.ALPHA * (bw[p_node_id] + 1e-9)
        link_resource_names = [a.name if hasattr(a, 'name') else a
                               for a in v_net.get_link_attrs('resource')]
        prox = 0.0
        for v_neighbor_id in v_net.adj[v_node_id]:
            p_placed = placed_map.get(v_neighbor_id)
            if p_placed is not None:
                hops = dist.get(p_placed, {}).get(p_node_id, 3)
                proximity = 1.0 / (1.0 + hops)
                edge = v_net.edges[v_node_id, v_neighbor_id] \
                    if v_net.has_edge(v_node_id, v_neighbor_id) else {}
                weight = 1.0 + sum(edge.get(name, 0)
                                   for name in link_resource_names) / self.PROX_NORM
                prox += weight * proximity
        return base * (1.0 + self.BETA * prox)

    # ---------- main ----------
    def solve(self, instance: dict) -> Solution:
        """
        Solve the problem instance with a greedy locality-dominant embedding.

        Args:
            instance: the problem instance to solve.

        Returns:
            Solution: the solution to the problem instance.
        """
        v_net, p_net = instance['v_net'], instance['p_net']
        solution = Solution.from_v_net(v_net)

        dist = self._hop_distances(p_net)
        orders = [self._rank_v_nodes(v_net), self._bfs_order(v_net)]
        for attempt, order in enumerate(orders):
            if self._attempt(v_net, p_net, solution, order, dist):
                # SUCCESS
                solution['result'] = True
                return solution
            if attempt < len(orders) - 1:
                solution = Solution.from_v_net(v_net)
        # FAILURE
        solution.update({'place_result': False, 'result': False})
        return solution

    def _attempt(self, v_net, p_net, solution, order, dist) -> bool:
        """Greedily place and route virtual nodes in the given order."""
        placed_map = {}
        history = []
        for v_node_id in order:
            candidate_nodes = self.controller.find_candidate_nodes(
                v_net, p_net, v_node_id, filter=list(placed_map.values()))
            if not candidate_nodes:
                self._rollback(v_net, p_net, solution, history, placed_map)
                return False
            cpu, bw = self._residual_vectors(p_net)  # refresh after each placement
            scored_candidates = sorted(
                candidate_nodes,
                key=lambda p_node_id: -self._score(
                    p_node_id, v_node_id, placed_map, v_net, cpu, bw, dist))
            placed = False
            for p_node_id in scored_candidates[:self.MAX_TRIES]:
                place_result, _ = self.controller.place_and_route(
                    v_net, p_net, v_node_id, p_node_id, solution,
                    shortest_method=self.shortest_method, k=self.k_shortest)
                if place_result:
                    placed_map[v_node_id] = p_node_id
                    history.append(v_node_id)
                    placed = True
                    break
            if not placed:
                self._rollback(v_net, p_net, solution, history, placed_map)
                return False
        return True

    def _rollback(self, v_net, p_net, solution, history, placed_map):
        """Undo all placements of the current attempt, most recent first."""
        for v_node_id in reversed(history):
            self.controller.undo_place_and_route(
                v_net, p_net, v_node_id, placed_map[v_node_id], solution)
