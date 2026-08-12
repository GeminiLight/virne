from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from omegaconf import OmegaConf

from virne.core import Solution
from virne.network import (
    AttributeBenchmarkManager,
    AttributeBenchmarks,
    PhysicalNetwork,
    TopologicalMetricCalculator,
    VirtualNetwork,
)
from virne.solver.learning.obs_handler import ObservationHandler
from virne.solver.learning.utils import get_unexistent_link_pairs
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.buffer import RolloutBuffer
from virne.solver.learning.rl_core.instance_rl_environment import (
    InstanceRLEnv,
    JointPRStepInstanceRLEnv,
    NodePairStepInstanceRLEnv,
    NodeSlotsStepInstanceRLEnv,
)
from virne.solver.learning.rl_core.rl_enviroment_base import RLBaseEnv
from virne.solver.learning.rl_core.rl_solver import A2CSolver, PPOSolver, RLSolver


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


def make_p_net(node_ids=(0, 1)):
    p_net = PhysicalNetwork(
        config={
            'node_attrs_setting': NODE_ATTRS,
            'link_attrs_setting': LINK_ATTRS,
        }
    )
    p_net.add_nodes_from((node_id, {'cpu': 10.0}) for node_id in node_ids)
    if len(node_ids) > 1:
        p_net.add_edge(node_ids[0], node_ids[1], bw=10.0)
    return p_net


def make_v_net(node_ids=(0,)):
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
    v_net.add_nodes_from((node_id, {'cpu': 1.0}) for node_id in node_ids)
    if len(node_ids) > 1:
        v_net.add_edge(node_ids[0], node_ids[1], bw=1.0)
    v_net.ranked_nodes = list(node_ids)
    return v_net


class MinimalJointEnv(JointPRStepInstanceRLEnv):
    def get_observation(self):
        return {'action_mask': self.generate_action_mask()}

    def compute_reward(self, *args, **kwargs):
        return 0.0


def make_minimal_joint_env(candidate_nodes, p_node_ids=(0, 1), v_node_ids=(0,)):
    env = MinimalJointEnv.__new__(MinimalJointEnv)
    env.p_net = make_p_net(p_node_ids)
    env.v_net = make_v_net(v_node_ids)
    env.controller = Mock()
    env.controller.find_candidate_nodes.return_value = list(candidate_nodes)
    env.counter = Mock()
    env.counter.count_solution.side_effect = lambda _v_net, solution: solution.to_dict()
    env.solution = Solution.from_v_net(env.v_net)
    env.reusable = False
    env.shortest_method = 'k_shortest'
    env.k_shortest = 10
    RLBaseEnv.__init__(env)
    return env


def make_feature_config(schema_version, feature_name='p_net_v_node'):
    return OmegaConf.create({
        'rl': {
            'feature_constructor': {
                'schema_version': schema_version,
                'name': feature_name,
                'extracted_attr_types': ['resource'],
                'if_use_node_status_flags': True,
                'if_use_aggregated_link_attrs': True,
                'if_use_degree_metric': False,
                'if_use_more_topological_metrics': False,
            },
        },
    })


def cache_test_benchmarks():
    AttributeBenchmarkManager.clear_cache()
    AttributeBenchmarkManager.add_to_cache(
        'p_net',
        AttributeBenchmarks(
            node_attr_benchmarks={'cpu': 10.0},
            link_attr_benchmarks={'bw': 10.0},
            link_sum_attr_benchmarks={'bw': 10.0},
        ),
    )


def test_no_feasible_action_terminates_without_controller_mutation():
    env = make_minimal_joint_env(candidate_nodes=[])

    observation = env.get_observation()
    assert observation['action_mask'].sum() == 1

    _, _, terminated, truncated, info = env.step(
        int(np.flatnonzero(observation['action_mask'])[0])
    )

    assert terminated is True
    assert truncated is False
    assert env.solution.place_result is False
    assert info['description'] == 'No feasible physical node'
    env.controller.place_and_route.assert_not_called()


def test_revoked_action_filter_ignores_actions_not_in_current_candidates():
    env = make_minimal_joint_env(candidate_nodes=[0])
    env.allow_revocable = True
    env.revocable_action = env.p_net.num_nodes
    env.num_actions = env.p_net.num_nodes + 1
    env.revoked_actions_dict[(str(env.solution.node_slots), env.curr_v_node_id)].append(1)

    mask = env.generate_action_mask()

    assert mask.tolist() == [True, False, False]


def test_first_failed_action_does_not_try_to_revoke_empty_solution():
    env = make_minimal_joint_env(candidate_nodes=[0])
    env.allow_revocable = True
    env.revocable_action = env.p_net.num_nodes
    env.num_actions = env.p_net.num_nodes + 1
    env.controller.place_and_route.return_value = (False, {})

    _, _, terminated, truncated, _ = env.step(0)

    assert terminated is True
    assert truncated is False
    assert env.solution.revoke_times == 0


def test_joint_environment_fixed_action_trace_preserves_mapping_results():
    env = make_minimal_joint_env(
        candidate_nodes=[0, 1],
        v_node_ids=(0, 1),
    )
    env.p_net_backup = env.p_net.copy()

    def find_candidate_nodes(_v_net, _p_net, _v_node_id, filter):
        return [node_id for node_id in (0, 1) if node_id not in filter]

    def place_and_route(_v_net, _p_net, v_node_id, p_node_id, solution, **kwargs):
        solution.node_slots[v_node_id] = p_node_id
        return True, {}

    env.controller.find_candidate_nodes.side_effect = find_candidate_nodes
    env.controller.place_and_route.side_effect = place_and_route
    env.counter.count_partial_solution.side_effect = (
        lambda _v_net, solution: solution.to_dict()
    )

    observation, info = env.reset(seed=7)
    assert info == {}
    np.testing.assert_array_equal(observation['action_mask'], [True, True])

    observation, reward, terminated, truncated, _ = env.step(1)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    np.testing.assert_array_equal(observation['action_mask'], [True, False])

    observation, reward, terminated, truncated, _ = env.step(0)
    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert env.solution.result is True
    assert env.solution.node_slots == {0: 1, 1: 0}
    assert env.solution.selected_actions == [1, 0]
    np.testing.assert_array_equal(observation['action_mask'], [True, False])


def test_terminal_gae_characterization_values_are_stable():
    buffer = RolloutBuffer()
    buffer.add({}, 0, 1.0, False, 0.0, value=0.5)
    buffer.add({}, 1, 2.0, True, 0.0, value=1.0)

    buffer.compute_returns_and_advantages(
        last_value=0.0,
        gamma=0.9,
        gae_lambda=0.8,
        method='gae',
    )

    np.testing.assert_allclose(buffer.advantages, [2.12, 1.0])
    np.testing.assert_allclose(buffer.returns, [2.62, 2.0])


def test_a2c_entropy_is_a_bonus_when_minimizing_the_loss():
    solver = A2CSolver.__new__(A2CSolver)
    solver.buffer = RolloutBuffer()
    solver.buffer.observations = [{}, {}]
    solver.buffer.actions = [0, 1]
    solver.buffer.returns = [1.0, 2.0]
    solver.preprocess_obs = lambda observations, device: observations
    solver.device = torch.device('cpu')
    solver.evaluate_actions = Mock(return_value=(
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.4, 0.6]),
        {},
    ))
    solver.coef_critic_loss = 0.5
    solver.coef_entropy_loss = 0.01
    solver.config = OmegaConf.create({
        'rl': {'norm_advantage': False},
        'training': {'distributed_training': False},
    })
    solver.update_grad = Mock(return_value=torch.tensor(0.0))
    solver.optimizer = SimpleNamespace(defaults={'lr': 0.001})
    solver.logger = Mock()
    solver.lr_scheduler = None
    solver.update_time = 0

    solver.update()

    minimized_loss = solver.update_grad.call_args.args[0]
    assert minimized_loss.item() == pytest.approx(-0.005)


def make_minimal_ppo_solver(current_values):
    class PolicyThatMustNotBeCopied:
        def __deepcopy__(self, memo):
            raise AssertionError('PPO update must not copy an unused old policy')

    solver = PPOSolver.__new__(PPOSolver)
    solver.buffer = RolloutBuffer()
    for index, (rollout_value, return_value) in enumerate(
        zip([1.0, 2.0], [3.0, 5.0], strict=True)
    ):
        solver.buffer.add(
            {'index': index},
            0,
            0.0,
            True,
            np.array([0.0], dtype=np.float32),
            value=rollout_value,
        )
        solver.buffer.returns.append(return_value)
    solver.preprocess_obs = lambda observations, device: observations
    solver.device = torch.device('cpu')
    solver.policy = PolicyThatMustNotBeCopied()
    solver.evaluate_actions = Mock(return_value=(
        torch.tensor(current_values, dtype=torch.float32),
        torch.tensor([0.2, -0.1]),
        torch.zeros(2),
        {},
    ))
    solver.batch_size = 2
    solver.repeat_times = 0
    solver.eps_clip = 0.2
    solver.coef_critic_loss = 0.0
    solver.coef_entropy_loss = 0.0
    solver.coef_mask_loss = 0.0
    solver.criterion_critic = torch.nn.MSELoss()
    solver.config = OmegaConf.create({
        'rl': {
            'norm_reward': False,
            'norm_advantage': False,
            'target_kl': None,
        },
        'training': {
            'distributed_training': False,
            'log_interval': 1,
        },
    })
    solver.update_grad = Mock(return_value=torch.tensor(0.0))
    solver.logger = Mock()
    solver.lr_scheduler = None
    solver.update_time = 0
    return solver


def test_ppo_actor_advantages_do_not_drift_with_current_critic(monkeypatch):
    monkeypatch.setattr(
        torch,
        'randint',
        lambda *args, **kwargs: torch.tensor([0, 1]),
    )
    actor_losses = []
    for current_values in ([10.0, 20.0], [-10.0, -20.0]):
        solver = make_minimal_ppo_solver(current_values)

        solver.update()

        actor_losses.append(solver.logger.log.call_args.kwargs['data']['loss/actor_loss'])

    np.testing.assert_allclose(actor_losses[0], actor_losses[1])


@pytest.mark.parametrize(
    ('buffer_size', 'batch_size', 'repeat_times', 'expected_updates'),
    [
        (128, 128, 10, 10),
        (256, 128, 10, 20),
        (129, 128, 10, 11),
        (32, 128, 0, 1),
    ],
)
def test_ppo_update_count_uses_ceiling_without_an_extra_divisible_batch(
    buffer_size,
    batch_size,
    repeat_times,
    expected_updates,
):
    solver = PPOSolver.__new__(PPOSolver)
    solver.buffer = Mock()
    solver.buffer.size.return_value = buffer_size
    solver.batch_size = batch_size
    solver.repeat_times = repeat_times

    assert solver.calculate_update_sample_times() == expected_updates


def test_sync_vector_environment_uses_gymnasium_reset_and_step_contracts():
    def make_env():
        env = make_minimal_joint_env(candidate_nodes=[])
        env.p_net_backup = env.p_net.copy()
        env.observation_space = spaces.Dict({
            'action_mask': spaces.MultiBinary(env.num_actions),
        })
        return env

    vector_env = gym.vector.SyncVectorEnv([make_env, make_env])
    try:
        observations, infos = vector_env.reset(seed=11)
        np.testing.assert_array_equal(
            observations['action_mask'],
            [[True, False], [True, False]],
        )
        assert infos == {}

        _, _, terminated, truncated, _ = vector_env.step(np.array([0, 0]))
        np.testing.assert_array_equal(terminated, [True, True])
        np.testing.assert_array_equal(truncated, [False, False])
    finally:
        vector_env.close()


def test_instance_environment_passes_gymnasium_contract_check():
    env = make_minimal_joint_env(candidate_nodes=[])
    env.p_net_backup = env.p_net.copy()
    env.observation_space = spaces.Dict({
        'action_mask': spaces.MultiBinary(env.num_actions),
    })

    check_env(env, skip_render_check=True)


def test_empty_link_index_has_pyg_compatible_shape():
    link_index = ObservationHandler().get_link_index_obs(make_v_net())

    assert link_index.shape == (2, 0)
    assert link_index.dtype == np.int64


def test_isolated_v_node_link_aggregations_are_zero():
    v_net = make_v_net()
    handler = ObservationHandler()

    for aggregation in ('sum', 'mean', 'max', 'min'):
        values = handler.get_v_node_aggr_link_demands(
            v_net,
            0,
            aggr=aggregation,
            link_attr_types=['resource'],
        )
        np.testing.assert_array_equal(values, np.zeros(1, dtype=np.float32))


def test_equal_average_distances_normalize_to_finite_zeros():
    p_net = make_p_net()

    values = ObservationHandler().get_average_distance(
        p_net,
        nodes_slots={0: 0, 1: 1},
        normalization=True,
    )

    np.testing.assert_array_equal(values, np.zeros((2, 1), dtype=np.float32))
    assert np.isfinite(values).all()


def test_non_contiguous_node_ids_are_encoded_as_tensor_row_indices():
    p_net = make_p_net((10, 20))
    handler = ObservationHandler()

    np.testing.assert_array_equal(
        handler.get_link_index_obs(p_net),
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        handler.get_p_net_nodes_status(p_net, make_v_net(), {0: 20}),
        np.array([[0.0], [1.0]], dtype=np.float32),
    )
    assert get_unexistent_link_pairs(p_net).size == 0


def test_non_contiguous_physical_ids_use_dense_actions_and_real_node_ids():
    env = make_minimal_joint_env(candidate_nodes=[20], p_node_ids=(10, 20))
    def place_and_route(_v_net, _p_net, v_node_id, p_node_id, solution, **kwargs):
        solution.node_slots[v_node_id] = p_node_id
        return True, {}

    env.controller.place_and_route.side_effect = place_and_route

    mask = env.get_observation()['action_mask']
    assert mask.tolist() == [False, True]

    env.step(1)

    assert env.solution.selected_actions == [20]
    assert env.controller.place_and_route.call_args.args[3] == 20


def test_tuple_node_ids_use_dense_actions_and_tensor_indices():
    p_node_ids = ((0, 0), (0, 1))
    v_node_ids = (('left', 0), ('right', 1))
    env = make_minimal_joint_env(
        candidate_nodes=[p_node_ids[1]],
        p_node_ids=p_node_ids,
        v_node_ids=v_node_ids,
    )
    def place_and_route(_v_net, _p_net, v_node_id, p_node_id, solution, **kwargs):
        solution.node_slots[v_node_id] = p_node_id
        return True, {}

    env.controller.place_and_route.side_effect = place_and_route

    mask = env.get_observation()['action_mask']
    env.step(1)

    assert mask.tolist() == [False, True]
    assert env.curr_v_node_index == 1
    assert env.controller.place_and_route.call_args.args[2] == v_node_ids[0]
    assert env.controller.place_and_route.call_args.args[3] == p_node_ids[1]


def test_tuple_node_ids_construct_dense_dual_graph_features():
    p_net = make_p_net(((0, 0), (0, 1)))
    v_net = make_v_net((('left', 0), ('right', 1)))
    solution = Solution.from_v_net(v_net)
    cache_test_benchmarks()
    TopologicalMetricCalculator.clear_cache()
    config = make_feature_config(2, 'p_net_v_net')
    constructor = FeatureConstructorRegistry.get('p_net_v_net')(
        p_net,
        v_net,
        config,
    )

    observation = constructor.construct(p_net, v_net, solution, ('left', 0))

    np.testing.assert_array_equal(
        observation['p_net_edge_index'],
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        observation['v_net_edge_index'],
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    )


def test_rl_solver_rejects_special_actions_unsupported_by_policy_head(tmp_path):
    config = OmegaConf.create({
        'experiment': {
            'seed': 0,
            'run_id': 'unsupported-actions',
            'save_root_dir': str(tmp_path),
        },
        'solver': {
            'solver_name': 'test',
            'reusable': False,
            'node_ranking_method': 'order',
            'link_ranking_method': 'order',
            'matching_mathod': 'greedy',
            'shortest_method': 'k_shortest',
            'k_shortest': 10,
            'allow_rejection': True,
            'allow_revocable': False,
        },
    })

    with pytest.raises(NotImplementedError, match='one logit per physical node'):
        RLSolver(
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            config,
            make_policy=Mock(),
            obs_as_tensor=Mock(),
        )


def test_incomplete_composite_action_environments_fail_clearly():
    for env_cls in (NodePairStepInstanceRLEnv, NodeSlotsStepInstanceRLEnv):
        with pytest.raises(NotImplementedError, match='action head'):
            env_cls(None, None, None, None, None, None, None)


def test_terminal_rollout_uses_zero_bootstrap_without_evaluating_terminal_obs():
    class TerminalEnv:
        def __init__(self, p_net, v_net, *args):
            self.solution = Solution.from_v_net(v_net)
            self.curr_v_node_id = list(v_net.nodes)[0]

        def reset(self, *, seed=None, options=None):
            return {'value': 1.0}, {}

        def step(self, action):
            return {'value': np.nan}, 0.0, True, False, {}

    agent = InstanceAgent(TerminalEnv)
    agent.controller = Mock()
    agent.recorder = Mock()
    agent.counter = Mock()
    agent.logger = Mock()
    agent.config = OmegaConf.create({})
    agent.device = torch.device('cpu')
    agent.preprocess_obs = Mock(side_effect=lambda observation, device: observation)
    agent.select_action = Mock(return_value=(0, np.array(0.0)))
    agent.estimate_value = Mock(return_value=1.0)
    p_net = make_p_net()
    v_net = make_v_net()

    _, _, last_value = agent.learn_with_instance({'p_net': p_net, 'v_net': v_net})

    assert last_value == 0.0
    agent.estimate_value.assert_called_once_with({'value': 1.0})


def test_truncated_rollout_bootstraps_without_marking_transition_terminal():
    class TruncatedEnv:
        def __init__(self, p_net, v_net, *args):
            self.solution = Solution.from_v_net(v_net)
            self.curr_v_node_id = list(v_net.nodes)[0]

        def reset(self, *, seed=None, options=None):
            return {'value': 1.0}, {}

        def step(self, action):
            return {'value': 2.0}, 0.0, False, True, {}

    agent = InstanceAgent(TruncatedEnv)
    agent.controller = Mock()
    agent.recorder = Mock()
    agent.counter = Mock()
    agent.logger = Mock()
    agent.config = OmegaConf.create({})
    agent.device = torch.device('cpu')
    agent.preprocess_obs = Mock(side_effect=lambda observation, device: observation)
    agent.select_action = Mock(return_value=(0, np.array(0.0)))
    agent.estimate_value = Mock(
        side_effect=[torch.tensor([1.0]), torch.tensor([2.0])]
    )

    _, buffer, last_value = agent.learn_with_instance({
        'p_net': make_p_net(),
        'v_net': make_v_net(),
    })

    assert buffer.dones == [False]
    assert last_value == 2.0


def test_single_node_v_net_features_have_valid_empty_edges():
    p_net = make_p_net()
    v_net = make_v_net()
    solution = Solution.from_v_net(v_net)
    cache_test_benchmarks()
    TopologicalMetricCalculator.clear_cache()

    for feature_name in ('p_net_v_node', 'p_net_v_net'):
        config = make_feature_config(2, feature_name)
        constructor = FeatureConstructorRegistry.get(feature_name)(p_net, v_net, config)

        observation = constructor.construct(p_net, v_net, solution, 0)

        assert np.isfinite(observation['p_net_x']).all()
        if feature_name == 'p_net_v_net':
            assert observation['v_net_edge_index'].shape == (2, 0)
            assert observation['v_net_edge_attr'].shape == (0, 1)


def test_feature_schema_v2_corrects_mean_link_normalization():
    p_net = make_p_net()
    v_net = make_v_net((0, 1))
    solution = Solution.from_v_net(v_net)

    means = {}
    for schema_version in (1, 2):
        cache_test_benchmarks()
        TopologicalMetricCalculator.clear_cache()
        config = make_feature_config(schema_version)
        config.rl.feature_constructor.if_use_node_status_flags = False
        constructor = FeatureConstructorRegistry.get('p_net_v_node')(
            p_net,
            v_net,
            config,
        )
        observation = constructor.construct(p_net, v_net, solution, 0)
        # node resource, then min/mean/max/sum aggregated link resources
        means[schema_version] = observation['p_net_x'][:, 2]

    np.testing.assert_array_equal(means[1], np.array([5.0, 5.0]))
    np.testing.assert_array_equal(means[2], np.array([0.5, 0.5]))


def test_feature_schema_v2_uses_decision_progress_not_raw_virtual_id():
    v_net = make_v_net((5, 9))

    status = ObservationHandler().get_v_node_status(
        v_net,
        v_node_id=9,
        p_net_num_nodes=10,
        v_node_position=0,
        schema_version=2,
    )

    np.testing.assert_allclose(status, np.array([0.5, 0.2, 0.5], dtype=np.float32))


def test_instance_observation_uses_scalar_v_net_size_and_dense_v_node_index():
    env = InstanceRLEnv.__new__(InstanceRLEnv)
    env.p_net = make_p_net()
    env.v_net = make_v_net((5, 9))
    env.solution = Solution.from_v_net(env.v_net)
    env.feature_constructor = Mock()
    env.feature_constructor.construct.return_value = {}
    env.v_node_ids = list(env.v_net.nodes)
    env.v_node_id_to_index = {5: 0, 9: 1}
    env.p_node_ids = list(env.p_net.nodes)
    env.p_node_id_to_action = {0: 0, 1: 1}
    env.num_actions = 2
    env.allow_rejection = False
    env.allow_revocable = False
    env.revoked_actions_dict = {}
    env.controller = Mock()
    env.controller.find_candidate_nodes.return_value = [0, 1]

    observation = InstanceRLEnv.get_observation(env)

    assert observation['curr_v_node_id'] == 0
    assert observation['v_net_size'] == 2
    assert isinstance(observation['v_net_size'], int)


def make_checkpoint_solver(schema_version=2):
    solver = RLSolver.__new__(RLSolver)
    solver.policy = torch.nn.Linear(2, 1)
    solver.optimizer = torch.optim.Adam(solver.policy.parameters(), lr=0.01)
    solver.device = torch.device('cpu')
    solver.logger = Mock()
    solver.config = OmegaConf.create({
        'rl': {'feature_constructor': {'schema_version': schema_version}},
    })
    return solver


def test_legacy_checkpoint_selects_feature_schema_v1(tmp_path):
    source = make_checkpoint_solver()
    checkpoint_path = tmp_path / 'legacy.pkl'
    torch.save({
        'policy': source.policy.state_dict(),
        'optimizer': source.optimizer.state_dict(),
    }, checkpoint_path)
    target = make_checkpoint_solver(schema_version=2)

    target.load_model(checkpoint_path)

    assert target.config.rl.feature_constructor.schema_version == 1
    target.logger.warning.assert_called_once()


def test_versioned_checkpoint_restores_its_feature_schema(tmp_path):
    source = make_checkpoint_solver(schema_version=2)
    checkpoint_path = tmp_path / 'versioned.pkl'
    torch.save({
        'checkpoint_version': 2,
        'feature_schema_version': 2,
        'policy': source.policy.state_dict(),
        'optimizer': source.optimizer.state_dict(),
    }, checkpoint_path)
    target = make_checkpoint_solver(schema_version=1)

    target.load_model(checkpoint_path)

    assert target.config.rl.feature_constructor.schema_version == 2
    target.logger.warning.assert_not_called()


def test_incompatible_checkpoint_raises_instead_of_using_random_weights(tmp_path):
    checkpoint_path = tmp_path / 'invalid.pkl'
    torch.save({'unexpected': torch.ones(1)}, checkpoint_path)
    solver = make_checkpoint_solver()

    with pytest.raises(RuntimeError, match='Load pretrained failed'):
        solver.load_model(checkpoint_path)
