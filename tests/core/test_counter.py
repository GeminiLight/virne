from collections import OrderedDict

import numpy as np
import pytest
from omegaconf import OmegaConf

from virne.core import Counter, Solution
from virne.network import VirtualNetwork


NODE_ATTRS = [
    {
        'name': 'cpu',
        'type': 'resource',
        'owner': 'node',
        'generative': False,
    },
]
LINK_ATTRS = [
    {
        'name': 'bw',
        'type': 'resource',
        'owner': 'link',
        'generative': False,
    },
]


def make_v_net():
    v_net = VirtualNetwork(
        config={
            'node_attrs_setting': NODE_ATTRS,
            'link_attrs_setting': LINK_ATTRS,
            'graph_attrs_setting': {
                'id': 0,
                'arrival_time': 1.0,
                'lifetime': 4.0,
            },
        }
    )
    v_net.add_nodes_from([(0, {'cpu': 2.0}), (1, {'cpu': 3.0})])
    v_net.add_edge(0, 1, bw=4.0)
    return v_net


def make_success_solution(v_net):
    solution = Solution.from_v_net(v_net)
    solution.result = True
    solution.node_slots = OrderedDict([(0, 0), (1, 2)])
    solution.node_slots_info = OrderedDict([
        ((0, 0), {'cpu': 2.0}),
        ((1, 2), {'cpu': 3.0}),
    ])
    solution.link_paths = OrderedDict([
        ((0, 1), [(0, 1), (1, 2)]),
    ])
    solution.link_paths_info = OrderedDict([
        (((0, 1), (0, 1)), {'bw': 4.0}),
        (((0, 1), (1, 2)), {'bw': 4.0}),
    ])
    return solution


def make_counter(metric_schema_version=None):
    config = {}
    if metric_schema_version is not None:
        config['metrics'] = {'schema_version': metric_schema_version}
    return Counter(NODE_ATTRS, LINK_ATTRS, {}, OmegaConf.create(config))


def test_count_solution_total_demand_is_complete_and_idempotent():
    counter = make_counter()
    v_net = make_v_net()
    solution = make_success_solution(v_net)

    first = counter.count_solution(v_net, solution)
    second = counter.count_solution(v_net, solution)

    assert first['v_net_node_demand'] == 5.0
    assert first['v_net_link_demand'] == 4.0
    assert first['v_net_demand'] == 9.0
    assert first['v_net_time_rc_ratio'] == pytest.approx(first['v_net_r2c_ratio'])
    assert second['v_net_demand'] == 9.0


def test_partial_and_complete_multi_resource_accounting_are_consistent():
    multi_node_attrs = [
        {
            'name': name,
            'type': 'resource',
            'owner': 'node',
            'generative': False,
        }
        for name in ['cpu', 'ram']
    ]
    v_net = VirtualNetwork(
        config={
            'node_attrs_setting': multi_node_attrs,
            'link_attrs_setting': LINK_ATTRS,
            'graph_attrs_setting': {
                'id': 0,
                'arrival_time': 0.0,
                'lifetime': 2.0,
            },
        }
    )
    v_net.add_node(0, cpu=2.0, ram=4.0)
    counter = Counter(multi_node_attrs, LINK_ATTRS, {}, OmegaConf.create({}))

    partial = Solution.from_v_net(v_net)
    partial.node_slots[0] = 0
    partial_info = counter.count_partial_solution(v_net, partial)

    complete = Solution.from_v_net(v_net)
    complete.result = True
    complete.node_slots[0] = 0
    complete.node_slots_info[(0, 0)] = {'cpu': 2.0, 'ram': 4.0}
    complete_info = counter.count_solution(v_net, complete)

    assert partial_info['v_net_node_revenue'] == 3.0
    assert partial_info['v_net_revenue'] == 3.0
    assert partial_info['v_net_r2c_ratio'] == 1.0
    assert complete_info['v_net_revenue'] == 3.0
    assert complete_info['v_net_r2c_ratio'] == 1.0


def test_metric_schema_v2_sums_multi_resource_node_demand_consistently():
    multi_node_attrs = [
        {
            'name': name,
            'type': 'resource',
            'owner': 'node',
            'generative': False,
        }
        for name in ['cpu', 'ram']
    ]
    v_net = VirtualNetwork(
        config={
            'node_attrs_setting': multi_node_attrs,
            'link_attrs_setting': LINK_ATTRS,
            'graph_attrs_setting': {
                'id': 0,
                'arrival_time': 0.0,
                'lifetime': 2.0,
            },
        }
    )
    v_net.add_node(0, cpu=2.0, ram=4.0)
    counter = Counter(
        multi_node_attrs,
        LINK_ATTRS,
        {},
        OmegaConf.create({'metrics': {'schema_version': 2}}),
    )
    solution = Solution.from_v_net(v_net)
    solution.result = True
    solution.node_slots[0] = 0
    solution.node_slots_info[(0, 0)] = {'cpu': 2.0, 'ram': 4.0}

    complete_info = counter.count_solution(v_net, solution)

    assert complete_info['metric_schema_version'] == 2
    assert complete_info['v_net_node_demand'] == 6.0
    assert complete_info['v_net_revenue'] == 6.0
    assert complete_info['v_net_cost'] == 6.0
    assert counter.calculate_v_net_revenue(v_net) == 6.0
    assert counter.calculate_v_net_cost(v_net, solution) == 6.0


def test_legacy_metric_schema_helpers_match_recorded_multi_resource_values():
    multi_node_attrs = [
        {
            'name': name,
            'type': 'resource',
            'owner': 'node',
            'generative': False,
        }
        for name in ['cpu', 'ram']
    ]
    v_net = VirtualNetwork(
        config={
            'node_attrs_setting': multi_node_attrs,
            'link_attrs_setting': LINK_ATTRS,
            'graph_attrs_setting': {
                'id': 0,
                'arrival_time': 0.0,
                'lifetime': 2.0,
            },
        }
    )
    v_net.add_node(0, cpu=2.0, ram=4.0)
    counter = Counter(multi_node_attrs, LINK_ATTRS, {}, OmegaConf.create({}))
    solution = Solution.from_v_net(v_net)
    solution.result = True
    solution.node_slots[0] = 0
    solution.node_slots_info[(0, 0)] = {'cpu': 2.0, 'ram': 4.0}

    complete_info = counter.count_solution(v_net, solution)

    assert complete_info['metric_schema_version'] == 1
    assert complete_info['v_net_revenue'] == 3.0
    assert counter.calculate_v_net_revenue(v_net) == 3.0
    assert counter.calculate_v_net_cost(v_net, solution) == 3.0


def test_unsupported_metric_schema_fails_clearly():
    with pytest.raises(ValueError, match='Unsupported metric schema version'):
        make_counter(metric_schema_version=3)


def test_equal_revenue_and_cost_with_float_resources_does_not_exceed_one():
    rng = np.random.default_rng(0)
    rng.uniform(-12, 12, size=10)
    link_demands = 10.0 ** rng.uniform(-12, 12, size=10)
    v_net = VirtualNetwork(
        config={
            'node_attrs_setting': NODE_ATTRS,
            'link_attrs_setting': LINK_ATTRS,
            'graph_attrs_setting': {
                'id': 0,
                'arrival_time': 0.0,
                'lifetime': 1.0,
            },
        }
    )
    solution = Solution.from_v_net(v_net)
    solution.result = True
    for node_id in range(20):
        v_net.add_node(node_id, cpu=0.0)
        solution.node_slots[node_id] = node_id
        solution.node_slots_info[(node_id, node_id)] = {'cpu': 0.0}
    for edge_id, link_demand in enumerate(link_demands):
        v_link = (2 * edge_id, 2 * edge_id + 1)
        v_net.add_edge(*v_link, bw=float(link_demand))
        solution.link_paths[v_link] = [v_link]
        solution.link_paths_info[(v_link, v_link)] = {'bw': float(link_demand)}

    info = make_counter().count_solution(v_net, solution)

    assert info['v_net_r2c_ratio'] == 1.0


def test_summary_uses_event_horizon_and_safe_zero_ratios():
    records = [
        {
            'event_type': event_type,
            'event_time': event_time,
            'success_count': 0,
            'v_net_count': 1,
            'v_net_r2c_ratio': 0.0,
            'total_time_revenue': 0.0,
            'total_time_cost': 0.0,
            'early_rejection': True,
            'place_result': True,
            'route_result': True,
            'total_cost': 0.0,
            'total_revenue': 0.0,
            'v_net_arrival_time': 1.0,
            'p_net_available_resource': 50.0,
            'p_net_node_available_resource': 30.0,
            'p_net_link_available_resource': 20.0,
            'inservice_count': 0,
            'v_net_total_hard_constraint_violation': 0.0,
            'v_net_max_single_step_hard_constraint_violation': 0.0,
            'v_net_reward': 0.0,
        }
        for event_type, event_time in [(1, 1.0), (0, 101.0)]
    ]

    summary = make_counter().summary_records(records)

    assert summary['metric_schema_version'] == 1
    assert summary['acceptance_rate'] == 0.0
    assert summary['failure_count'] == 1
    assert summary['early_rejection_count'] == 1
    assert summary['unknown_failure_count'] == 0
    assert summary['long_term_time_r2c_ratio'] == 0.0
    assert summary['long_term_r2c_ratio'] == 0.0
    assert summary['total_simulation_time'] == 101.0
    assert summary['long_term_avg_time_revenue'] == 0.0
    assert summary['long_term_avg_revenue'] == 0.0
    assert summary['long_term_avg_cost'] == 0.0


def test_summary_uses_last_event_time_for_average_revenue():
    records = [
        {
            'event_type': event_type,
            'event_time': event_time,
            'success_count': 1,
            'v_net_count': 1,
            'v_net_r2c_ratio': 9.0 / 13.0,
            'total_time_revenue': 36.0,
            'total_time_cost': 52.0,
            'early_rejection': False,
            'place_result': True,
            'route_result': True,
            'total_cost': 13.0,
            'total_revenue': 9.0,
            'v_net_arrival_time': 1.0,
            'p_net_available_resource': 50.0,
            'p_net_node_available_resource': 30.0,
            'p_net_link_available_resource': 20.0,
            'inservice_count': int(event_type == 1),
            'v_net_total_hard_constraint_violation': 0.0,
            'v_net_max_single_step_hard_constraint_violation': 0.0,
            'v_net_reward': 0.0,
        }
        for event_type, event_time in [(1, 1.0), (0, 5.0)]
    ]

    summary = make_counter().summary_records(records)

    assert summary['total_simulation_time'] == 5.0
    assert summary['long_term_avg_time_revenue'] == pytest.approx(36.0 / 5.0)
    assert summary['long_term_avg_revenue'] == pytest.approx(9.0 / 5.0)
    assert summary['long_term_avg_cost'] == pytest.approx(13.0 / 5.0)


def test_summary_counts_constraint_violations_on_arrival_only():
    records = [
        {
            'event_type': event_type,
            'event_time': event_time,
            'success_count': 0,
            'v_net_count': 1,
            'v_net_r2c_ratio': 0.0,
            'total_time_revenue': 0.0,
            'total_time_cost': 0.0,
            'early_rejection': False,
            'place_result': True,
            'route_result': True,
            'total_cost': 0.0,
            'total_revenue': 0.0,
            'v_net_arrival_time': 1.0,
            'p_net_available_resource': 50.0,
            'p_net_node_available_resource': 30.0,
            'p_net_link_available_resource': 20.0,
            'inservice_count': 0,
            'v_net_total_hard_constraint_violation': 2.0,
            'v_net_max_single_step_hard_constraint_violation': 1.0,
        }
        for event_type, event_time in [(1, 1.0), (0, 5.0)]
    ]

    summary = make_counter().summary_records(records)

    assert summary['total_violation'] == 2.0
    assert summary['total_max_single_step_violation'] == 1.0
