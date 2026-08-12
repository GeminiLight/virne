from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from virne.solver.learning.rl_core.buffer import RolloutBuffer
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.rl_solver import DDPGSolver, DQNSolver


class TinyActionPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(1, 1)

    def act(self, observation):
        return self.actor(observation['features']).squeeze(-1)


def make_off_policy_solver(solver_cls, algorithm_name):
    solver = solver_cls.__new__(solver_cls)
    solver.algorithm_name = algorithm_name
    solver.config = OmegaConf.create(
        {
            'training': {
                'distributed_training': False,
                'log_interval': 1,
            },
            'rl': {
                'gamma': 0.9,
                'mask_actions': True,
                'clip_grad': True,
                'max_grad_norm': 1.0,
                'weight_decay': 0.0,
                'learning_rate': {'actor': 0.01, 'critic': 0.01},
            },
        },
    )
    solver.device = torch.device('cpu')
    solver.batch_size = 2
    solver.buffer = RolloutBuffer()
    solver.policy = TinyActionPolicy()
    solver.optimizer = torch.optim.Adam(solver.policy.parameters(), lr=0.01)
    solver.preprocess_obs = lambda observations, device: {
        'features': torch.as_tensor(
            np.stack([observation['features'] for observation in observations]),
            dtype=torch.float32,
            device=device,
        ),
        'action_mask': torch.as_tensor(
            np.stack([observation['action_mask'] for observation in observations]),
            dtype=torch.bool,
            device=device,
        ),
    }
    solver.logger = SimpleNamespace(log=lambda *args, **kwargs: None)
    solver.update_time = 0
    solver.epsilon = 0.0
    return solver


def add_transitions(solver):
    transitions = [
        (
            {'features': np.array([[1.0], [2.0]]), 'action_mask': np.array([True, True])},
            0,
            1.0,
            False,
            {'features': np.array([[2.0], [1.0]]), 'action_mask': np.array([False, True])},
        ),
        (
            {'features': np.array([[2.0], [1.0]]), 'action_mask': np.array([True, True])},
            1,
            -0.5,
            True,
            {'features': np.array([[3.0], [4.0]]), 'action_mask': np.array([True, True])},
        ),
    ]
    for observation, action, reward, done, next_observation in transitions:
        solver.buffer.add(
            observation,
            action,
            reward,
            done,
            np.zeros((1, 1), dtype=np.float32),
            value=0.0,
            next_obs=next_observation,
        )


def test_rollout_buffer_trim_preserves_newest_aligned_transitions():
    buffer = RolloutBuffer()
    for index in range(4):
        buffer.add(index, index, index, False, index, index, index + 1)

    buffer.trim(2)

    assert buffer.size() == 2
    assert buffer.curr_idx == 2
    assert buffer.observations == [2, 3]
    assert buffer.actions == [2, 3]
    assert buffer.next_observations == [3, 4]


def test_off_policy_merge_keeps_failed_experience_without_gae():
    agent = InstanceAgent.__new__(InstanceAgent)
    agent.is_off_policy = True
    agent.replay_capacity = 2
    agent.buffer = RolloutBuffer()
    agent.time_step = 0
    instance_buffer = RolloutBuffer()
    for index in range(3):
        instance_buffer.add(index, index, -1.0, True, 0.0, next_obs=index + 1)

    agent.merge_instance_experience(
        instance=None,
        solution={'result': False},
        instance_buffer=instance_buffer,
        last_value=0.0,
    )

    assert agent.buffer.observations == [1, 2]
    assert agent.buffer.returns == []
    assert agent.time_step == 1


def test_dqn_exploration_samples_only_legal_actions():
    solver = make_off_policy_solver(DQNSolver, 'dqn')
    solver.epsilon_start = 1.0
    solver.epsilon_end = 1.0
    solver.epsilon_decay = 1.0
    solver.steps_done = 0
    observation = {
        'features': torch.tensor([[[1.0], [2.0], [3.0]]]),
        'action_mask': torch.tensor([[False, True, False]]),
    }

    actions = [solver.select_action(observation, sample=True)[0] for _ in range(20)]

    assert actions == [1] * 20


def test_dqn_update_uses_replay_without_clearing_it():
    solver = make_off_policy_solver(DQNSolver, 'dqn')
    solver.target_policy = deepcopy(solver.policy)
    solver.target_update_interval = 1
    add_transitions(solver)
    old_parameters = [parameter.detach().clone() for parameter in solver.policy.parameters()]

    loss = solver.update()

    assert torch.isfinite(loss)
    assert solver.buffer.size() == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(old_parameters, solver.policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(online, target)
        for online, target in zip(
            solver.policy.state_dict().values(),
            solver.target_policy.state_dict().values(),
            strict=True,
        )
    )


def test_dqn_checkpoint_metadata_is_safe_for_weights_only_loading(tmp_path):
    solver = make_off_policy_solver(DQNSolver, 'dqn')
    solver.target_policy = deepcopy(solver.policy)
    solver.steps_done = 3
    solver.epsilon = np.float64(0.25)
    checkpoint_path = tmp_path / 'dqn.pkl'

    torch.save(solver.get_additional_checkpoint_state(), checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint['steps_done'] == 3
    assert checkpoint['epsilon'] == 0.25


def test_discrete_ddpg_updates_actor_and_critic_and_soft_targets():
    solver = make_off_policy_solver(DDPGSolver, 'ddpg')
    solver.critic_policy = TinyActionPolicy()
    solver.critic_optimizer = torch.optim.Adam(
        solver.critic_policy.parameters(),
        lr=0.01,
    )
    solver.target_policy = deepcopy(solver.policy)
    solver.target_critic_policy = deepcopy(solver.critic_policy)
    solver.tau = 0.2
    solver.policy_temperature = 1.0
    add_transitions(solver)
    actor_before = [parameter.detach().clone() for parameter in solver.policy.parameters()]
    critic_before = [parameter.detach().clone() for parameter in solver.critic_policy.parameters()]

    loss = solver.update()

    assert torch.isfinite(loss)
    assert solver.buffer.size() == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, solver.policy.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, solver.critic_policy.parameters(), strict=True)
    )


@pytest.mark.parametrize('solver_name', ['dqn_mlp+', 'ddpg_mlp+'])
def test_off_policy_mlp_solvers_are_registered(solver_name):
    from virne.solver import SolverRegistry

    assert SolverRegistry.get(solver_name).type == 'r_learning'
