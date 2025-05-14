# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import numpy as np
from gym import spaces
from typing import Any, Dict, Tuple, List, Union, Optional, Type, Callable


import torch
import numpy as np
from omegaconf import open_dict

from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger

from virne.solver import SolverRegistry
from virne.solver.learning.rl_policy import AttActorCritic
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry, PNetVNodeFeatureConstructor, PNetVNetFeatureConstructor
from virne.solver.learning.rl_core.reward_calculator import RewardCalculatorRegistry, VanillaRewardCalculator, AdaptiveWeightRewardCalculator
from virne.solver.learning.reinforcement_learning.solver_maker import make_solver_class


obs_as_tensor = TensorConvertor.obs_as_tensor_for_att
build_policy = PolicyBuilder.build_att_policy


class AttInstanceRLEnv(PlaceStepInstanceRLEnv):
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(AttInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
        self.feature_constructor = FeatureConstructorRegistry.get(config.rl.feature_constructor.name)(
            self.node_attr_benchmarks or {}, self.link_attr_benchmarks or {}, self.link_sum_attr_benchmarks or {}, self.config)

class PpoAttInstanceRLEnv(PlaceStepInstanceRLEnv):
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(PpoAttInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
            config.rl.feature_constructor.if_use_degree_metric = True
            config.rl.feature_constructor.if_use_more_topological_metrics = False
            config.rl.feature_constructor.if_use_aggregated_link_attrs = True
            config.rl.feature_constructor.if_use_node_status_flags = True
            config.rl.reward_calculator.name = 'adaptive_intermediate'
            config.rl.if_use_negative_sample = False
        self.feature_constructor = PNetVNetFeatureConstructor(self.node_attr_benchmarks or {}, self.link_attr_benchmarks or {}, self.link_sum_attr_benchmarks or {}, self.config)
        self.reward_calculator = AdaptiveWeightRewardCalculator(self.config)


# @SolverRegistry.register(solver_name='ppo_att', solver_type='r_learning')
@SolverRegistry.register(solver_name='ppo_att', solver_type='r_learning')
class PpoAttSolver(InstanceAgent, PGSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Convolutional Neural Network (CNN) as the neural network model.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, PpoAttInstanceRLEnv)
        PGSolver.__init__(self, controller, recorder, counter, logger, config, build_policy, obs_as_tensor, **kwargs)


extended_att_solvers = [
    {'solver_name': 'pg_att+', 'solver_cls_name': 'PgAttSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_att+', 'solver_cls_name': 'A2cAttSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_att+', 'solver_cls_name': 'A3cAttSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_att+', 'solver_cls_name': 'PpoAttSolver', 'rl_solver_cls': PPOSolver},
]
for solver_info in extended_att_solvers:
    # make and register the solver class
    solver_name = solver_info['solver_name']
    policy_builder = build_policy
    obs_as_tensor = TensorConvertor.obs_as_tensor_for_att
    instance_env_cls = AttInstanceRLEnv
    base_solver_cls = solver_info['rl_solver_cls']
    make_solver_class(solver_name, instance_env_cls, base_solver_cls, policy_builder, obs_as_tensor)

