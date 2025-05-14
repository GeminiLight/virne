from typing import Callable, Dict, Any, Optional, Type
from abc import ABC, abstractmethod
import numpy as np
from virne.core import Solution
from virne.network import VirtualNetwork, PhysicalNetwork


class BaseRewardCalculator(ABC):
    """
    Abstract base class for reward calculation. Users can extend this for custom reward logic.
    To use a custom reward, subclass this and register your class with RewardCalculatorRegistry.
    """
    def __init__(self, config):
        self.config = config
        self.v_net_reward = 0.0
        intermediate_reward = self.config.rl.reward_calculator.intermediate_reward
        assert intermediate_reward == -1 or intermediate_reward >= 0, 'intermediate_reward should be a non-negative number or -1'


    @abstractmethod
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        pass

class RewardCalculatorRegistry:
    """
    Registry for reward calculator classes. Supports registration and retrieval by name.
    """
    _registry: Dict[str, Type[BaseRewardCalculator]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(calculator_cls: Type[BaseRewardCalculator]):
            if name in cls._registry:
                raise ValueError(f"Reward calculator '{name}' is already registered.")
            cls._registry[name] = calculator_cls
            return calculator_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseRewardCalculator]:
        if name not in cls._registry:
            raise NotImplementedError(f"Reward calculator '{name}' is not implemented.")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[BaseRewardCalculator]]:
        return dict(cls._registry)


@RewardCalculatorRegistry.register('gradual_intermediate')
class GradualIntermediateRewardCalculator(BaseRewardCalculator):
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        if solution.get('result', False):
            reward = float(solution.get('v_net_r2c_ratio', 0.0))
        elif solution.get('place_result', False) and solution.get('route_result', False):
            curr_place_progress = get_curr_place_progress(v_net, solution)
            v_net_r2c_ratio = float(solution.get('v_net_r2c_ratio', 0.0))
            reward = 0.1 * curr_place_progress * v_net_r2c_ratio
        else:
            reward = -float(get_curr_place_progress(v_net, solution))
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward

@RewardCalculatorRegistry.register('adaptive_intermediate')
class AdaptiveWeightRewardCalculator(BaseRewardCalculator):
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        weight = 1 / v_net.num_nodes
        if solution.get('result', False):
            reward = float(solution.get('v_net_r2c_ratio', 0.0))
        elif solution.get('place_result', False) and solution.get('route_result', False):
            reward = weight
        else:
            reward = -weight
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward


@RewardCalculatorRegistry.register('fixed_intermediate')
class FixedWeightRewardCalculator(BaseRewardCalculator):
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        weight = self.config.rl.reward_calculator.intermediate_reward
        if solution.get('result', False):
            reward = float(solution.get('v_net_r2c_ratio', 0.0))
        elif solution.get('place_result', False) and solution.get('route_result', False):
            reward = weight
        else:
            reward = -weight
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward


@RewardCalculatorRegistry.register('vanilla')
class VanillaRewardCalculator(BaseRewardCalculator):
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        if solution.get('result', False):
            reward = float(solution.get('v_net_r2c_ratio', 0.0))
        else:
            reward = 0.0
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward


def get_curr_place_progress(v_net: VirtualNetwork, solution: Solution) -> float:
    """Calculate the current placement progress."""
    node_slots = solution.get('node_slots', None)
    if node_slots is None or not hasattr(node_slots, 'keys'):
        return 0.0
    num_placed_v_net_nodes = len(node_slots.keys())
    total_place_count = v_net.num_nodes - 1 if v_net.num_nodes > 1 else 1
    return num_placed_v_net_nodes / total_place_count

def get_node_load_balance(p_net: VirtualNetwork, p_node_id: int):
    n_attrs = p_net.get_node_attrs(['resource'])
    if len(n_attrs) > 1:
        n_resources = np.array([p_net.nodes[p_node_id][n_attr.name] for n_attr in n_attrs])
        load_balance = np.std(n_resources)
    else:
        n_attr = p_net.get_node_attrs(['extrema'])[0]
        load_balance = p_net.nodes[p_node_id][n_attr.originator] / p_net.nodes[p_node_id][n_attr.name]
    return load_balance


# def compute_reward(self, solution):
#     """Calculate deserved reward according to the result of taking action."""
#     weight = (1 / self.v_net.num_nodes)
#     if solution['result']:
#         node_load_balance = self.get_node_load_balance(self.selected_p_net_nodes[-1])
#         reward = weight * (solution['v_net_r2c_ratio'] + 0.01 * node_load_balance) 
#         reward += solution['v_net_r2c_ratio']
#     elif solution['place_result'] and solution['route_result']:
#         node_load_balance = self.get_node_load_balance(self.selected_p_net_nodes[-1])
#         reward = weight * ((solution['v_net_r2c_ratio']) + 0.01 * node_load_balance)
#     else:
#         reward = - weight
#     # reward = reward * self.v_net.total_resource_demand / 500
#     self.solution['v_net_reward'] += reward
#     return reward
