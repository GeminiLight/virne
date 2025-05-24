# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import numpy as np
from gym import spaces
from typing import Any, Dict, Tuple, List, Union, Optional, Type, Callable


import torch
import numpy as np
from omegaconf import DictConfig, open_dict

from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger

from virne.solver import SolverRegistry
from virne.solver.learning.rl_policy import MlpActorCritic
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry, PNetVNodeFeatureConstructor
from virne.solver.learning.rl_core.reward_calculator import RewardCalculatorRegistry, VanillaRewardCalculator
from virne.solver.learning.reinforcement_learning.solver_maker import make_solver_class


obs_as_tensor = TensorConvertor.obs_as_tensor_for_mlp
build_policy = PolicyBuilder.build_mlp_policy


class PgMlpInstanceRLEnv(PlaceStepInstanceRLEnv):
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
            config.rl.feature_constructor.if_use_degree_metric = True
            config.rl.feature_constructor.if_use_more_topological_metrics = False
            config.rl.feature_constructor.if_use_aggregated_link_attrs = True
            config.rl.feature_constructor.if_use_node_status_flags = True
            config.rl.reward_calculator.name = 'vanilla'
            config.rl.if_use_negative_sample = False
        super(PgMlpInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


@SolverRegistry.register(solver_name='pg_mlp', solver_type='r_learning')
class PgMlpSolver(InstanceAgent, PGSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Convolutional Neural Network (CNN) as the neural network model.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, PgMlpInstanceRLEnv)
        PGSolver.__init__(self, controller, recorder, counter, logger, config, build_policy, obs_as_tensor, **kwargs)


class PgMlp2InstanceRLEnv(PlaceStepInstanceRLEnv):
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
            config.rl.feature_constructor.if_use_degree_metric = True
            config.rl.feature_constructor.if_use_more_topological_metrics = True
            config.rl.feature_constructor.if_use_aggregated_link_attrs = True
            config.rl.feature_constructor.if_use_node_status_flags = True
            config.rl.reward_calculator.name = 'vanilla'
        super(PgMlp2InstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)



@SolverRegistry.register(solver_name='pg_mlp2', solver_type='r_learning')
class PgMlp2Solver(InstanceAgent, PGSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Convolutional Neural Network (CNN) as the neural network model.
    Additional features are used for the second version.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, PgMlp2InstanceRLEnv)
        PGSolver.__init__(self, controller, recorder, counter, logger, config, build_policy, obs_as_tensor, **kwargs)


class MlpInstanceEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
        super(MlpInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


extended_mlp_solvers = [
    {'solver_name': 'pg_mlp+', 'policy_key': 'mlp', 'solver_cls_name': 'PgMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'ppo_mlp+', 'policy_key': 'mlp', 'solver_cls_name': 'PpoMlpSolver', 'rl_solver_cls': PPOSolver},
    {'solver_name': 'a2c_mlp+', 'policy_key': 'mlp', 'solver_cls_name': 'A2cMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_mlp+', 'policy_key': 'mlp', 'solver_cls_name': 'A3cMlpSolver', 'rl_solver_cls': A3CSolver}
]
for solver_info in extended_mlp_solvers:
    # make and register the solver class
    solver_name = solver_info['solver_name']
    policy_key = '_'.join(solver_info['solver_name'].split('_')[1:])[:-1]
    policy_builder = build_policy
    obs_as_tensor = TensorConvertor.obs_as_tensor_for_mlp
    instance_env_cls = MlpInstanceEnv
    base_solver_cls = solver_info['rl_solver_cls']
    make_solver_class(solver_name, instance_env_cls, base_solver_cls, policy_builder, obs_as_tensor)