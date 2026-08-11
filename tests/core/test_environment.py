import copy
import time
from collections import OrderedDict
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from virne.core import Controller, Counter, Recorder, Solution
from virne.core.environment import JointPRStepEnvironment, SolutionStepEnvironment
from virne.network import PhysicalNetwork, VirtualNetwork, VirtualNetworkRequestSimulator
from virne.network.virtual_network_request_simulator import VirtualNetworkEvent
from virne.system.base_system import TimeWindowSystem


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


def make_config(tmp_path):
    return OmegaConf.create({
        'experiment': {
            'run_id': 'environment-audit',
            'seed': 0,
            'save_root_dir': str(tmp_path),
            'if_load_v_nets': False,
        },
        'solver': {
            'solver_name': 'environment-audit',
            'node_ranking_method': 'order',
            'link_ranking_method': 'order',
            'matching_mathod': 'greedy',
            'shortest_method': 'k_shortest',
            'k_shortest': 10,
        },
        'simulation': {
            'p_net_dataset_dir': str(tmp_path / 'p_net'),
            'v_nets_dataset_dir': str(tmp_path / 'v_nets'),
            'v_sim_setting_num_node_resource_attrs': 1,
            'v_sim_setting_num_link_resource_attrs': 1,
        },
        'recorder': {
            'if_temp_save_records': False,
            'if_save_records': False,
            'record_dir_name': 'records',
            'summary_file_name': 'summary.csv',
        },
    })


def make_networks():
    p_net = PhysicalNetwork(
        config={
            'node_attrs_setting': NODE_ATTRS,
            'link_attrs_setting': LINK_ATTRS,
        }
    )
    p_net.add_nodes_from([
        (0, {'cpu': 10.0}),
        (1, {'cpu': 10.0}),
        (2, {'cpu': 10.0}),
    ])
    p_net.add_edges_from([
        (0, 1, {'bw': 10.0}),
        (1, 2, {'bw': 10.0}),
    ])

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
    v_net.ranked_nodes = [0, 1]
    return p_net, v_net


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


def make_environment(
    tmp_path,
    env_cls=SolutionStepEnvironment,
    **env_kwargs,
):
    config = make_config(tmp_path)
    p_net, v_net = make_networks()
    simulator = VirtualNetworkRequestSimulator(
        v_nets=[v_net],
        events=[
            VirtualNetworkEvent(id=0, type=1, v_net_id=0, time=1.0),
            VirtualNetworkEvent(id=1, type=0, v_net_id=0, time=5.0),
        ],
        v_sim_setting={},
    )
    counter = Counter(NODE_ATTRS, LINK_ATTRS, {}, config)
    controller = Controller(NODE_ATTRS, LINK_ATTRS, [], config.solver)
    recorder = Recorder(counter, config)
    logger = Mock()
    env = env_cls(
        p_net,
        simulator,
        controller,
        recorder,
        counter,
        logger,
        config,
        **env_kwargs,
    )
    recorder.count_init_p_net_info(env.p_net)
    env.num_processed_v_nets = 0
    env.start_run_time = time.time()
    env.ready(0)
    return env, v_net


def test_successful_lifecycle_restores_resources_and_clears_bookkeeping(tmp_path):
    env, v_net = make_environment(tmp_path)

    _, _, done, arrival_record = env.step(make_success_solution(v_net))

    assert done is True
    assert arrival_record['result'] is True
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net.nodes] == [10.0, 10.0, 10.0]
    assert [env.p_net.links[e]['bw'] for e in env.p_net.links] == [10.0, 10.0]
    assert env.recorder.state['inservice_count'] == 0
    assert env.recorder.state['num_running_p_net_nodes'] == 0
    assert env.recorder.get_running_p_net_nodes() == []
    assert env.recorder.memory[-1]['result'] is True
    assert env.recorder.memory[-1]['node_slots'] == {0: 0, 1: 2}
    assert env.recorder.memory[-1]['description'] == 'Leave Event'


def test_admission_control_uses_computed_r2c_ratio(tmp_path):
    env, v_net = make_environment(
        tmp_path,
        r2c_ratio_threshold=0.5,
        vn_size_threshold=1,
    )

    _, _, _, arrival_record = env.step(make_success_solution(v_net))

    assert arrival_record['v_net_r2c_ratio'] == pytest.approx(9.0 / 13.0)
    assert arrival_record['result'] is True
    assert arrival_record['description'] == 'Success'


def test_admission_rejection_has_explicit_reason(tmp_path):
    env, v_net = make_environment(
        tmp_path,
        r2c_ratio_threshold=0.8,
        vn_size_threshold=1,
    )

    _, _, _, arrival_record = env.step(make_success_solution(v_net))

    assert arrival_record['result'] is False
    assert arrival_record['early_rejection'] is True
    assert arrival_record['failure_reason'] == 'early_rejection'
    assert arrival_record['description'] == 'Admission Rejection'
    assert env.recorder.state['success_count'] == 0


def test_constraint_rejection_has_explicit_reason(tmp_path):
    env, v_net = make_environment(tmp_path)
    solution = make_success_solution(v_net)
    solution.v_net_total_hard_constraint_violation = 1.0

    _, _, _, arrival_record = env.step(solution)

    assert arrival_record['result'] is False
    assert arrival_record['early_rejection'] is False
    assert arrival_record['failure_reason'] == 'constraint'
    assert arrival_record['description'] == 'Constraint Violation'
    summary = env.recorder.summary_records(env.recorder.memory)
    assert summary['failure_count'] == 1
    assert summary['constraint_failure_count'] == 1
    assert summary['unknown_failure_count'] == 0
    assert summary['total_violation'] == 1.0


def test_unclassified_solver_failure_is_counted_as_unknown(tmp_path):
    env, v_net = make_environment(tmp_path)

    _, _, _, arrival_record = env.step(Solution.from_v_net(v_net))

    assert arrival_record['result'] is False
    assert arrival_record['failure_reason'] == 'unknown'
    summary = env.recorder.summary_records(env.recorder.memory)
    assert summary['failure_count'] == 1
    assert summary['unknown_failure_count'] == 1
    assert (
        summary['early_rejection_count']
        + summary['constraint_failure_count']
        + summary['place_failure_count']
        + summary['route_failure_count']
        + summary['unknown_failure_count']
    ) == summary['failure_count']


def test_zero_capacity_resource_dimensions_have_zero_utilization(tmp_path):
    config = make_config(tmp_path)
    p_net = PhysicalNetwork(
        config={
            'node_attrs_setting': NODE_ATTRS,
            'link_attrs_setting': LINK_ATTRS,
        }
    )
    p_net.add_node(0, cpu=0.0)
    v_net = VirtualNetwork(
        config={
            'node_attrs_setting': NODE_ATTRS,
            'link_attrs_setting': LINK_ATTRS,
            'graph_attrs_setting': {
                'id': 0,
                'arrival_time': 1.0,
                'lifetime': 1.0,
            },
        }
    )
    v_net.add_node(0, cpu=0.0)
    counter = Counter(NODE_ATTRS, LINK_ATTRS, {}, config)
    recorder = Recorder(counter, config)
    recorder.count_init_p_net_info(p_net)
    recorder.update_state({'event_id': 0, 'event_type': 1, 'event_time': 1.0})

    record = recorder.count(v_net, p_net, Solution.from_v_net(v_net))

    assert record['p_net_node_resource_utilization'] == 0.0
    assert record['p_net_link_resource_utilization'] == 0.0


def test_equal_time_departure_releases_resources_before_next_arrival(tmp_path):
    env, first_v_net = make_environment(tmp_path)
    for node_id in first_v_net.nodes:
        first_v_net.nodes[node_id]['cpu'] = 10.0
    first_v_net.links[(0, 1)]['bw'] = 10.0
    first_v_net.arrival_time = 1.0
    first_v_net.lifetime = 4.0

    second_v_net = copy.deepcopy(first_v_net)
    second_v_net.id = 1
    second_v_net.arrival_time = 5.0
    second_v_net.lifetime = 4.0
    simulator = VirtualNetworkRequestSimulator(
        v_nets=[first_v_net, second_v_net],
        v_sim_setting={},
    )
    simulator._renew_events()
    env.v_net_simulator = simulator
    env.recorder.reset()
    env.recorder.count_init_p_net_info(env.p_net)
    env.num_processed_v_nets = 0
    env.ready(0)

    solution = make_success_solution(first_v_net)
    solution.node_slots_info[(0, 0)] = {'cpu': 10.0}
    solution.node_slots_info[(1, 2)] = {'cpu': 10.0}
    solution.link_paths_info[((0, 1), (0, 1))] = {'bw': 10.0}
    solution.link_paths_info[((0, 1), (1, 2))] = {'bw': 10.0}

    _, _, done, _ = env.step(solution)

    assert done is False
    assert env.curr_event.type == 1
    assert env.curr_event.v_net_id == 1
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net.nodes] == [10.0, 10.0, 10.0]
    assert [env.p_net.links[e]['bw'] for e in env.p_net.links] == [10.0, 10.0]


def test_noncontiguous_event_and_v_net_ids_do_not_control_list_positions(tmp_path):
    env, v_net = make_environment(tmp_path)
    v_net.id = 100
    env.v_net_simulator = VirtualNetworkRequestSimulator(
        v_nets=[v_net],
        events=[
            VirtualNetworkEvent(id=10, type=1, v_net_id=100, time=1.0),
            VirtualNetworkEvent(id=20, type=0, v_net_id=100, time=5.0),
        ],
        v_sim_setting={},
    )
    env.recorder.reset()
    env.recorder.count_init_p_net_info(env.p_net)
    env.num_processed_v_nets = 0
    env.ready(0)

    _, _, done, _ = env.step(make_success_solution(v_net))

    assert done is True
    assert len(env.recorder.memory) == 2
    assert env.recorder.get_record(event_id=10)['event_type'] == 1
    assert env.recorder.get_record(event_id=20)['event_type'] == 0
    assert env.recorder.get_record(v_net_id=100)['event_type'] == 1
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net.nodes] == [10.0, 10.0, 10.0]


def test_float_resource_accounting_is_tolerant_and_restores_resources(tmp_path):
    env, v_net = make_environment(tmp_path)
    v_net.nodes[0]['cpu'] = 0.1
    v_net.nodes[1]['cpu'] = 0.2
    v_net.links[(0, 1)]['bw'] = 0.1
    solution = make_success_solution(v_net)
    solution.node_slots_info[(0, 0)] = {'cpu': 0.1}
    solution.node_slots_info[(1, 2)] = {'cpu': 0.2}
    solution.link_paths_info[((0, 1), (0, 1))] = {'bw': 0.1}
    solution.link_paths_info[((0, 1), (1, 2))] = {'bw': 0.1}

    _, _, done, arrival_record = env.step(solution)

    assert done is True
    assert arrival_record['result'] is True
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net.nodes] == pytest.approx([10.0, 10.0, 10.0])
    assert [env.p_net.links[e]['bw'] for e in env.p_net.links] == pytest.approx([10.0, 10.0])


def test_metric_schema_v2_deploys_and_restores_multi_resource_demands(tmp_path):
    multi_node_attrs = [
        {
            'name': name,
            'type': 'resource',
            'owner': 'node',
            'generative': False,
        }
        for name in ['cpu', 'ram']
    ]
    config = make_config(tmp_path)
    config.metrics = {'schema_version': 2}
    config.simulation.v_sim_setting_num_node_resource_attrs = 2
    p_net = PhysicalNetwork(
        config={
            'node_attrs_setting': multi_node_attrs,
            'link_attrs_setting': LINK_ATTRS,
        }
    )
    p_net.add_nodes_from([
        (0, {'cpu': 10.0, 'ram': 10.0}),
        (1, {'cpu': 10.0, 'ram': 10.0}),
        (2, {'cpu': 10.0, 'ram': 10.0}),
    ])
    p_net.add_edges_from([
        (0, 1, {'bw': 10.0}),
        (1, 2, {'bw': 10.0}),
    ])
    v_net = VirtualNetwork(
        config={
            'node_attrs_setting': multi_node_attrs,
            'link_attrs_setting': LINK_ATTRS,
            'graph_attrs_setting': {
                'id': 0,
                'arrival_time': 1.0,
                'lifetime': 4.0,
            },
        }
    )
    v_net.add_nodes_from([
        (0, {'cpu': 2.0, 'ram': 4.0}),
        (1, {'cpu': 3.0, 'ram': 1.0}),
    ])
    v_net.add_edge(0, 1, bw=4.0)
    simulator = VirtualNetworkRequestSimulator(v_nets=[v_net], v_sim_setting={})
    simulator._renew_events()
    counter = Counter(multi_node_attrs, LINK_ATTRS, {}, config)
    controller = Controller(multi_node_attrs, LINK_ATTRS, [], config.solver)
    recorder = Recorder(counter, config)
    env = SolutionStepEnvironment(
        p_net,
        simulator,
        controller,
        recorder,
        counter,
        Mock(),
        config,
    )
    recorder.count_init_p_net_info(env.p_net)
    env.num_processed_v_nets = 0
    env.start_run_time = time.time()
    env.ready(0)
    solution = Solution.from_v_net(v_net)
    solution.result = True
    solution.node_slots = OrderedDict([(0, 0), (1, 2)])
    solution.node_slots_info = OrderedDict([
        ((0, 0), {'cpu': 2.0, 'ram': 4.0}),
        ((1, 2), {'cpu': 3.0, 'ram': 1.0}),
    ])
    solution.link_paths = OrderedDict([((0, 1), [(0, 1), (1, 2)])])
    solution.link_paths_info = OrderedDict([
        (((0, 1), (0, 1)), {'bw': 4.0}),
        (((0, 1), (1, 2)), {'bw': 4.0}),
    ])

    _, _, done, arrival_record = env.step(solution)

    assert done is True
    assert arrival_record['metric_schema_version'] == 2
    assert arrival_record['v_net_node_cost'] == 10.0
    assert arrival_record['v_net_revenue'] == 14.0
    assert arrival_record['v_net_cost'] == 18.0
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net.nodes] == [10.0, 10.0, 10.0]
    assert [env.p_net.nodes[n]['ram'] for n in env.p_net.nodes] == [10.0, 10.0, 10.0]


def test_invalid_solver_solution_is_rejected_before_mutating_p_net(tmp_path):
    env, v_net = make_environment(tmp_path)
    before = copy.deepcopy(env.p_net)
    solution = make_success_solution(v_net)
    solution.node_slots_info[(1, 2)] = {'cpu': 99.0}

    with pytest.raises(ValueError, match='node_slots_info'):
        env.step(solution)

    assert len(env.recorder.memory) == 0
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net.nodes] == [before.nodes[n]['cpu'] for n in before.nodes]
    assert [env.p_net.links[e]['bw'] for e in env.p_net.links] == [before.links[e]['bw'] for e in before.links]


def test_post_deploy_accounting_failure_rolls_back_p_net(tmp_path):
    env, v_net = make_environment(tmp_path)
    before = copy.deepcopy(env.p_net)

    def inconsistent_deploy(v_net, p_net, solution):
        p_net.nodes[0]['cpu'] -= 1.0
        return True

    env.controller.deploy = Mock(side_effect=inconsistent_deploy)

    with pytest.raises(ValueError, match='resource accounting mismatch'):
        env.step(make_success_solution(v_net))

    assert len(env.recorder.memory) == 0
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net.nodes] == [before.nodes[n]['cpu'] for n in before.nodes]
    assert [env.p_net.links[e]['bw'] for e in env.p_net.links] == [before.links[e]['bw'] for e in before.links]


def test_reset_uses_config_seed_and_clears_episode_extra_state(tmp_path, monkeypatch):
    env, _ = make_environment(tmp_path)
    env.config.experiment.seed = 7
    env.extra_record_info['one_off'] = 1
    env.extra_summary_info['one_off'] = 1
    monkeypatch.setattr(
        'virne.core.environment.get_v_nets_dataset_dir_from_setting',
        lambda setting, seed: str(tmp_path / f'missing-seed-{seed}'),
    )
    env.v_net_simulator.renew = Mock()

    env.reset()

    assert env.seed == 7
    env.v_net_simulator.renew.assert_called_once_with(v_nets=True, events=True, seed=7)
    assert env.extra_record_info == {}
    assert env.extra_summary_info == {}


def test_save_summary_does_not_mutate_input(tmp_path):
    env, _ = make_environment(tmp_path)
    env.config.rl = OmegaConf.create({
        'reward_calculator': {'name': 'vanilla', 'intermediate_reward': -1},
        'mask_actions': True,
        'feature_constructor': {
            'if_use_node_status_flags': False,
            'if_use_aggregated_link_attrs': False,
            'if_use_degree_metric': False,
            'if_use_more_topological_metrics': False,
        },
    })
    env.config.training = OmegaConf.create({'num_train_epochs': 0})
    summary = {'solver_name': 'solver', 'run_id': 'run', 'metric': 1.0}
    expected = copy.deepcopy(summary)

    env.recorder.save_summary(summary)

    assert summary == expected


def test_save_summary_preserves_legacy_aggregate_files(tmp_path):
    env, _ = make_environment(tmp_path)
    env.config.rl = OmegaConf.create({
        'reward_calculator': {'name': 'vanilla', 'intermediate_reward': -1},
        'mask_actions': True,
        'feature_constructor': {
            'if_use_node_status_flags': False,
            'if_use_aggregated_link_attrs': False,
            'if_use_degree_metric': False,
            'if_use_more_topological_metrics': False,
        },
    })
    env.config.training = OmegaConf.create({'num_train_epochs': 0})
    solver_summary_path = tmp_path / 'environment-audit' / 'solver_summary.csv'
    global_summary_path = tmp_path / 'global_summary.csv'
    solver_summary_path.write_text('legacy_field\nlegacy_value\n')
    global_summary_path.write_text('legacy_field\nlegacy_value\n')
    summary = {
        'solver_name': 'environment-audit',
        'run_id': 'environment-audit',
        'metric_schema_version': 1,
    }

    env.recorder.save_summary(summary)

    assert solver_summary_path.read_text() == 'legacy_field\nlegacy_value\n'
    assert global_summary_path.read_text() == 'legacy_field\nlegacy_value\n'
    assert (
        tmp_path / 'environment-audit' / 'solver_summary-metrics-v1.csv'
    ).exists()
    assert (tmp_path / 'global_summary-metrics-v1.csv').exists()


def test_time_window_system_fails_fast_with_actionable_message():
    with pytest.raises(NotImplementedError, match='system.if_time_window=false'):
        TimeWindowSystem(None, None, None, None, None, None, OmegaConf.create({}))


def test_reusable_mapping_mode_remains_explicitly_unsupported(tmp_path):
    config = make_config(tmp_path)
    config.solver.reusable = True
    p_net, v_net = make_networks()
    simulator = VirtualNetworkRequestSimulator(
        v_nets=[v_net],
        events=[VirtualNetworkEvent(id=0, type=1, v_net_id=0, time=1.0)],
        v_sim_setting={},
    )
    counter = Counter(NODE_ATTRS, LINK_ATTRS, {}, config)
    controller = Controller(NODE_ATTRS, LINK_ATTRS, [], config.solver)
    recorder = Recorder(counter, config)

    with pytest.raises(NotImplementedError, match='reusable=true'):
        SolutionStepEnvironment(
            p_net,
            simulator,
            controller,
            recorder,
            counter,
            Mock(),
            config,
        )


def test_joint_pr_environment_can_finish_and_record_solution(tmp_path):
    env, _ = make_environment(tmp_path, env_cls=JointPRStepEnvironment)

    def place_and_route(v_net, p_net, v_node_id, p_node_id, solution, **kwargs):
        solution.node_slots[v_node_id] = p_node_id
        solution.node_slots_info[(v_node_id, p_node_id)] = {
            'cpu': v_net.nodes[v_node_id]['cpu'],
        }
        if len(solution.node_slots) == v_net.num_nodes:
            solution.link_paths[(0, 1)] = [(0, 1), (1, 2)]
            solution.link_paths_info[((0, 1), (0, 1))] = {'bw': 4.0}
            solution.link_paths_info[((0, 1), (1, 2))] = {'bw': 4.0}
        return True, {}

    env.controller.place_and_route = Mock(side_effect=place_and_route)
    env.controller.release = Mock(return_value=True)

    _, _, first_done, _ = env.step(0)
    _, _, second_done, arrival_record = env.step(2)

    assert first_done is False
    assert second_done is True
    assert arrival_record['result'] is True
    assert arrival_record['node_slots'] == {0: 0, 1: 2}
    assert len(env.recorder.memory) == 2
