import copy
import math
from unittest.mock import Mock

from omegaconf import OmegaConf

from virne.core.controller.controller import Controller
from virne.core.solution import Solution
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.solver.heuristic.bfs_trials import OrderRankBfsSolver
from virne.solver.heuristic.node_rank import OrderRankSolver, PLRankSolver


HARD_NODE_ATTRS = [
    {'name': 'cpu', 'type': 'resource', 'owner': 'node', 'generative': False},
]
HARD_LINK_ATTRS = [
    {'name': 'bw', 'type': 'resource', 'owner': 'link', 'generative': False},
]
SOFT_NODE_ATTRS = [
    {
        'name': 'cpu',
        'type': 'resource',
        'owner': 'node',
        'generative': False,
        'constraint_restrictions': 'soft',
    },
]
SOFT_LINK_ATTRS = [
    {
        'name': 'bw',
        'type': 'resource',
        'owner': 'link',
        'generative': False,
        'constraint_restrictions': 'soft',
    },
]


def make_solver_config(
    solver_name='order_rank',
    matching_method='greedy',
    shortest_method='k_shortest',
    k_shortest=10,
):
    return OmegaConf.create(
        {
            'experiment': {'seed': 0, 'run_id': 'test', 'save_root_dir': '/tmp'},
            'solver': {
                'solver_name': solver_name,
                'reusable': False,
                'node_ranking_method': 'order',
                'link_ranking_method': 'order',
                'matching_mathod': matching_method,
                'shortest_method': shortest_method,
                'k_shortest': k_shortest,
                'allow_rejection': False,
                'allow_revocable': False,
            },
        }
    )


def make_controller(node_attrs=HARD_NODE_ATTRS, link_attrs=HARD_LINK_ATTRS, config=None):
    return Controller(
        node_attrs,
        link_attrs,
        [],
        config or {
            'reusable': False,
            'matching_mathod': 'greedy',
            'shortest_method': 'k_shortest',
        },
    )


def make_p_net(nodes, edges, node_attrs=HARD_NODE_ATTRS, link_attrs=HARD_LINK_ATTRS):
    net_config = {'node_attrs_setting': node_attrs, 'link_attrs_setting': link_attrs}
    p_net = PhysicalNetwork(config=net_config)
    p_net.add_nodes_from((node_id, {'cpu': float(cpu)}) for node_id, cpu in nodes)
    p_net.add_edges_from((u, v, {'bw': float(bw)}) for u, v, bw in edges)
    return p_net


def make_v_net(nodes, edges, node_attrs=HARD_NODE_ATTRS, link_attrs=HARD_LINK_ATTRS, v_net_id=1):
    net_config = {'node_attrs_setting': node_attrs, 'link_attrs_setting': link_attrs}
    v_net = VirtualNetwork(
        config=net_config,
        id=v_net_id,
        arrival_time=0.0,
        lifetime=1.0,
    )
    v_net.add_nodes_from((node_id, {'cpu': float(cpu)}) for node_id, cpu in nodes)
    v_net.add_edges_from((u, v, {'bw': float(bw)}) for u, v, bw in edges)
    return v_net


def resource_snapshot(p_net):
    return (
        {node_id: dict(p_net.nodes[node_id]) for node_id in p_net.nodes},
        {tuple(link): dict(p_net.links[link]) for link in p_net.links},
    )


def make_solver(solver_cls, config, controller=None):
    controller = controller or make_controller(config=config)
    return solver_cls(controller, Mock(), Mock(), Mock(), config)


def test_solver_and_controller_respect_nested_solver_config():
    config = make_solver_config(
        matching_method='l2s2',
        shortest_method='first_shortest',
        k_shortest=1,
    )
    controller = make_controller(config=config)
    solver = make_solver(OrderRankSolver, config, controller)
    bfs_solver = make_solver(OrderRankBfsSolver, config, controller)

    assert controller.matching_mathod == 'l2s2'
    assert controller.shortest_method == 'first_shortest'
    assert solver.matching_mathod == 'l2s2'
    assert solver.shortest_method == 'first_shortest'
    assert solver.k_shortest == 1
    assert bfs_solver.matching_mathod == 'l2s2'
    assert bfs_solver.shortest_method == 'first_shortest'
    assert bfs_solver.k_shortest == 1

    p_net = make_p_net([(0, 1), (1, 2), (2, 2)], [(0, 1, 10), (1, 2, 10), (0, 2, 10)])
    v_net = make_v_net([(0, 2), (1, 2)], [(0, 1, 1)])
    before = resource_snapshot(p_net)
    solution = solver.solve({'p_net': p_net, 'v_net': v_net})
    assert solution.result is False
    assert resource_snapshot(p_net) == before

    path_config = make_solver_config(
        matching_method='greedy',
        shortest_method='first_shortest',
        k_shortest=1,
    )
    path_solver = make_solver(OrderRankSolver, path_config)
    p_net = make_p_net([(0, 10), (1, 10), (2, 10)], [(0, 1, 2), (0, 2, 10), (2, 1, 10)])
    v_net = make_v_net([(0, 2), (1, 2)], [(0, 1, 4)])
    before = resource_snapshot(p_net)
    solution = path_solver.solve({'p_net': p_net, 'v_net': v_net})
    assert solution.result is False
    assert resource_snapshot(p_net) == before


def test_node_mapping_failure_rolls_back_and_handles_exhausted_candidates():
    controller = make_controller()
    p_net = make_p_net([(0, 10), (1, 1)], [(0, 1, 10)])
    v_net = make_v_net([(0, 2), (1, 2)], [(0, 1, 1)])
    before = resource_snapshot(p_net)
    solution = Solution.from_v_net(v_net)

    result = controller.node_mapper.node_mapping(
        v_net,
        p_net,
        [0, 1],
        [0, 1],
        solution,
        reusable=False,
        inplace=True,
        matching_mathod='greedy',
    )

    assert result is False
    assert resource_snapshot(p_net) == before
    assert solution.node_slots == {}

    p_net = make_p_net([(0, 10)], [])
    before = resource_snapshot(p_net)
    solution = Solution.from_v_net(v_net)
    result = controller.node_mapper.node_mapping(
        v_net,
        p_net,
        [0, 1],
        [0],
        solution,
        reusable=False,
        inplace=True,
        matching_mathod='greedy',
    )
    assert result is False
    assert resource_snapshot(p_net) == before
    assert solution.node_slots == {}


def test_link_mapping_and_node_slots_failure_roll_back_all_resources():
    controller = make_controller()
    p_net = make_p_net([(0, 10), (1, 10), (2, 10)], [(0, 1, 10), (1, 2, 3)])
    v_net = make_v_net([(0, 2), (1, 2), (2, 2)], [(0, 1, 4), (1, 2, 4)])
    before = resource_snapshot(p_net)
    solution = Solution.from_v_net(v_net)
    solution.node_slots.update({0: 0, 1: 1, 2: 2})

    result = controller.link_mapper.link_mapping(
        v_net,
        p_net,
        solution,
        sorted_v_links=[(0, 1), (1, 2)],
        shortest_method='k_shortest',
        inplace=True,
    )

    assert result is False
    assert resource_snapshot(p_net) == before
    assert solution.link_paths == {}

    solution = Solution.from_v_net(v_net)
    controller.deploy_with_node_slots(
        v_net,
        p_net,
        {0: 0, 1: 1, 2: 2},
        solution,
        inplace=True,
        shortest_method='k_shortest',
    )
    assert solution.result is False
    assert resource_snapshot(p_net) == before
    assert solution.node_slots == {}
    assert solution.link_paths == {}


def test_place_and_route_failure_rolls_back_only_the_current_step():
    controller = make_controller()
    p_net = make_p_net([(0, 10), (1, 10)], [(0, 1, 2)])
    v_net = make_v_net([(0, 1), (1, 1)], [(0, 1, 5)])
    solution = Solution.from_v_net(v_net)

    first_result, _ = controller.place_and_route(
        v_net, p_net, 0, 0, solution, shortest_method='k_shortest'
    )
    assert first_result is True
    before_second_step = resource_snapshot(p_net)

    second_result, _ = controller.place_and_route(
        v_net, p_net, 1, 1, solution, shortest_method='k_shortest'
    )

    assert second_result is False
    assert resource_snapshot(p_net) == before_second_step
    assert solution.node_slots == {0: 0}
    assert solution.link_paths == {}


def test_bulk_unsafe_mapping_and_node_slots_record_violations():
    controller = make_controller()
    p_net = make_p_net([(0, 2), (1, 2)], [(0, 1, 2)])
    v_net = make_v_net([(0, 5), (1, 5)], [(0, 1, 5)])
    solution = Solution.from_v_net(v_net)

    controller.deploy_with_node_slots(
        v_net,
        p_net,
        {0: 0, 1: 1},
        solution,
        inplace=True,
        if_allow_constraint_violation=True,
    )

    assert solution.result is True
    assert solution.is_feasible() is False
    assert solution.v_net_total_hard_constraint_violation == 9.0
    assert solution.v_net_max_single_step_hard_constraint_violation == 3.0
    assert p_net.nodes[0]['cpu'] == -3.0
    assert p_net.nodes[1]['cpu'] == -3.0
    assert p_net.links[(0, 1)]['bw'] == -3.0

    disconnected = make_p_net([(0, 2), (1, 2)], [])
    before = resource_snapshot(disconnected)
    failed_solution = Solution.from_v_net(v_net)
    controller.deploy_with_node_slots(
        v_net,
        disconnected,
        {0: 0, 1: 1},
        failed_solution,
        inplace=True,
        if_allow_constraint_violation=True,
    )
    assert failed_solution.result is False
    assert resource_snapshot(disconnected) == before
    assert failed_solution.node_slots == {}
    assert failed_solution.link_paths == {}


def test_soft_only_constraints_do_not_crash_when_recording_offsets():
    node_controller = make_controller(SOFT_NODE_ATTRS, HARD_LINK_ATTRS)
    p_net = make_p_net([(0, 10), (1, 10)], [(0, 1, 10)], SOFT_NODE_ATTRS, HARD_LINK_ATTRS)
    v_net = make_v_net([(0, 1), (1, 1)], [(0, 1, 3)], SOFT_NODE_ATTRS, HARD_LINK_ATTRS)
    solution = Solution.from_v_net(v_net)

    node_result, _ = node_controller.node_mapper.place(v_net, p_net, 0, 0, solution)
    assert node_result is True
    assert solution.v_net_total_hard_constraint_violation == 0.0

    link_controller = make_controller(HARD_NODE_ATTRS, SOFT_LINK_ATTRS)
    p_net = make_p_net([(0, 10), (1, 10)], [(0, 1, 10)], HARD_NODE_ATTRS, SOFT_LINK_ATTRS)
    v_net = make_v_net([(0, 1), (1, 1)], [(0, 1, 3)], HARD_NODE_ATTRS, SOFT_LINK_ATTRS)
    solution = Solution.from_v_net(v_net)
    solution.node_slots.update({0: 0, 1: 1})
    link_result, _ = link_controller.link_mapper.route(
        v_net,
        p_net,
        (0, 1),
        (0, 1),
        solution,
        shortest_method='k_shortest',
    )
    assert link_result is True
    assert solution.v_net_total_hard_constraint_violation == 0.0


def test_bfs_path_boundaries_and_available_k_shortest():
    controller = make_controller()
    v_net = make_v_net([(0, 1), (1, 1)], [(0, 1, 1)])
    p_net = make_p_net([(10, 1), (20, 1), (30, 1)], [(10, 20, 10), (20, 30, 10)])

    assert controller.topology_analyzer.find_bfs_shortest_path(
        v_net, p_net, (0, 1), 10, 30
    ) == [10, 20, 30]
    assert controller.topology_analyzer.find_bfs_shortest_path(
        v_net, p_net, (0, 1), 10, 10
    ) == [10]

    disconnected = make_p_net([(0, 1), (1, 1)], [])
    assert controller.topology_analyzer.find_bfs_shortest_path(
        v_net, disconnected, (0, 1), 0, 1
    ) is None

    one_hop = make_p_net([(0, 1), (1, 1)], [(0, 1, 10)])
    assert controller.topology_analyzer.find_shortest_paths(
        v_net, one_hop, (0, 1), (0, 1), method='first_shortest', max_hop=1
    ) == [[0, 1]]
    assert controller.topology_analyzer.find_shortest_paths(
        v_net, one_hop, (0, 1), (0, 1), method='available_k_shortest', k=2
    ) == [[0, 1]]


def test_pl_uses_bfs_node_order_and_path_ranker():
    config = make_solver_config(solver_name='pl_rank')
    controller = make_controller(config=config)
    solver = make_solver(PLRankSolver, config, controller)

    p_net = make_p_net(
        [(0, 200), (1, 200), (2, 200)],
        [(0, 1, 20), (0, 2, 20), (1, 2, 20)],
    )
    v_net = make_v_net(
        [(0, 1), (1, 100), (2, 1)],
        [(0, 1, 1), (0, 2, 1)],
    )
    solution = Solution.from_v_net(v_net)
    assert solver.node_mapping(v_net, copy.deepcopy(p_net), solution) is True
    assert list(solution.node_slots) == [0, 1, 2]

    p_net = make_p_net(
        [(0, 10), (1, 100), (2, 1), (3, 10)],
        [(0, 1, 10), (1, 3, 10), (0, 2, 9), (2, 3, 9)],
    )
    v_net = make_v_net([(0, 1), (1, 1)], [(0, 1, 1)])
    solution = Solution.from_v_net(v_net)
    solution.node_slots.update({0: 0, 1: 3})
    rank_path = Mock(wraps=solver.rank_path)

    result = controller.link_mapper.link_mapping(
        v_net,
        p_net,
        solution,
        shortest_method='k_shortest',
        k=2,
        rank_path_func=rank_path,
    )

    assert result is True
    rank_path.assert_called_once()
    assert solution.link_paths[(0, 1)] == [(0, 2), (2, 3)]


def test_bulk_solution_violation_metrics_are_finite():
    config = make_solver_config()
    controller = make_controller(config=config)
    solver = make_solver(OrderRankSolver, config, controller)
    p_net = make_p_net([(0, 10), (1, 10)], [(0, 1, 10)])
    v_net = make_v_net([(0, 1), (1, 1)], [(0, 1, 2)])

    solution = solver.solve({'p_net': p_net, 'v_net': v_net})

    assert solution.result is True
    assert solution.v_net_single_step_hard_constraint_offset == 0.0
    assert solution.v_net_max_single_step_hard_constraint_violation == 0.0
    assert math.isfinite(solution.v_net_max_single_step_hard_constraint_violation)
