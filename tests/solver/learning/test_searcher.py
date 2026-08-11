from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from virne.solver.learning.rl_core.searcher import (
    BeamSearcher,
    RecoverableSearcher,
    SampleSearcher,
    get_searcher,
    select_action,
)


class FakeSolution(dict):
    def __init__(self):
        super().__init__(node_slots={}, result=False)
        self.result = False
        self.v_net_r2c_ratio = 0.0

    def is_feasible(self):
        return self.result

    @property
    def node_slots(self):
        return self['node_slots']


class TwoStepEnv:
    def __init__(self):
        self.path = []
        self.solution = FakeSolution()
        self.v_net = SimpleNamespace(num_nodes=2)
        self.logger = Mock()

    def get_observation(self):
        if not self.path:
            code = 0
        else:
            code = self.path[0] + 1
        return {
            'code': code,
            'action_mask': np.array([True, True]),
        }

    def generate_action_mask(self):
        return np.array([True, True])

    def step(self, action):
        self.path.append(int(action))
        done = len(self.path) == 2
        if done and self.path == [1, 0]:
            self.solution.result = True
            self.solution.v_net_r2c_ratio = 2.0
        self.solution['node_slots'] = dict(enumerate(self.path))
        return self.get_observation(), 0.0, done, False, {}


class RetryEnv(TwoStepEnv):
    def __init__(self):
        super().__init__()
        self.v_net = SimpleNamespace(num_nodes=1)
        self.curr_v_node_id = 0

    def step(self, action):
        self.solution.result = int(action) == 1
        self.solution['result'] = self.solution.result
        self.solution.v_net_r2c_ratio = 1.0 if self.solution.result else 0.0
        return self.get_observation(), 0.0, True, False, {}


class BranchPolicy(torch.nn.Module):
    def act(self, observation):
        code = int(observation['code'].reshape(-1)[0])
        if code == 2:
            return torch.tensor([[5.0, -5.0]])
        return torch.tensor([[2.0, 1.0]])


def preprocess(observation, device=None):
    return {
        'code': torch.tensor([observation['code']], device=device),
        'action_mask': torch.from_numpy(
            np.expand_dims(observation['action_mask'], axis=0),
        ).to(
            device=device,
        ),
    }


def test_all_advertised_searchers_accept_factory_arguments():
    policy = BranchPolicy()

    for strategy, k in (
        ('random', 1),
        ('greedy', 1),
        ('sample', 1),
        ('sample', 2),
        ('beam', 2),
        ('recoverable', 2),
        ('recovable', 2),
    ):
        searcher = get_searcher(
            strategy,
            policy=policy,
            preprocess_obs_func=preprocess,
            make_policy_func=Mock(),
            k=k,
            device=torch.device('cpu'),
            mask_actions=True,
            maskable_policy=True,
        )
        assert searcher.k == k


def test_unmaskable_policy_action_selection_uses_raw_distribution():
    action, log_prob = select_action(
        BranchPolicy(),
        {'code': torch.tensor([0])},
        mask=np.array([[True, False]]),
        sample=False,
        mask_actions=True,
        maskable_policy=False,
    )

    assert action == 0
    assert torch.isfinite(log_prob).all()


def test_sample_search_k_greater_than_one_uses_valid_worker_signature():
    searcher = SampleSearcher(
        BranchPolicy(),
        preprocess,
        make_policy_func=Mock(),
        k=2,
        device=torch.device('cpu'),
        parallel_searching=False,
    )

    solution = searcher.find_solution(TwoStepEnv())

    assert isinstance(solution, FakeSolution)


def test_beam_search_keeps_parent_state_aligned_after_branching():
    searcher = BeamSearcher(
        BranchPolicy(),
        preprocess,
        make_policy_func=Mock(),
        k=2,
        device=torch.device('cpu'),
        parallel_searching=False,
    )

    solution = searcher.find_solution(TwoStepEnv())

    assert solution.result is True
    assert solution['node_slots'] == {0: 1, 1: 0}


def test_recoverable_search_retries_a_failed_action():
    searcher = RecoverableSearcher(
        BranchPolicy(),
        preprocess,
        make_policy_func=Mock(),
        k=2,
        device=torch.device('cpu'),
        parallel_searching=False,
    )

    solution = searcher.find_solution(RetryEnv())

    assert solution.result is True
    assert solution['num_retry_times'] == 1
