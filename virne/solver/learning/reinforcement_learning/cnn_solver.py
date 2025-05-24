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
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder
from virne.solver.learning.reinforcement_learning.solver_maker import make_solver_class


obs_as_tensor = TensorConvertor.obs_as_tensor_for_cnn
build_policy = PolicyBuilder.build_cnn_policy


class PgCnnInstanceRLEnv(PlaceStepInstanceRLEnv):
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
            config.rl.feature_constructor.if_use_degree_metric = True
            config.rl.feature_constructor.if_use_more_topological_metrics = False
            config.rl.feature_constructor.if_use_aggregated_link_attrs = True
            config.rl.feature_constructor.if_use_node_status_flags = True
            config.rl.reward_calculator.name = 'vanilla'
            config.rl.if_use_negative_sample = False
        super(PgCnnInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


@SolverRegistry.register(solver_name='pg_cnn', solver_type='r_learning')
class PgCnnSolver(InstanceAgent, PGSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Convolutional Neural Network (CNN) as the neural network model.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, PgCnnInstanceRLEnv)
        PGSolver.__init__(self, controller, recorder, counter, logger, config, build_policy, obs_as_tensor, **kwargs)


class PgCnn2InstanceRLEnv(PlaceStepInstanceRLEnv):
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
            config.rl.feature_constructor.if_use_degree_metric = True
            config.rl.feature_constructor.if_use_more_topological_metrics = True
            config.rl.feature_constructor.if_use_aggregated_link_attrs = True
            config.rl.feature_constructor.if_use_node_status_flags = True
            config.rl.reward_calculator.name = 'vanilla'
        super(PgCnn2InstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


@SolverRegistry.register(solver_name='pg_cnn2', solver_type='r_learning')
class PgCnn2Solver(InstanceAgent, PGSolver):
    """
    A Reinforcement Learning-based solver that uses 
    Policy Gradient (PG) as the training algorithm and 
    Convolutional Neural Network (CNN) as the neural network model.
    Additionally, more graph features are used as the input of the CNN.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, PgCnn2InstanceRLEnv)
        PGSolver.__init__(self, controller, recorder, counter, logger, config, build_policy, obs_as_tensor, **kwargs)


class CnnInstanceEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
        super(CnnInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


extented_cnn_solvers = [
    {'solver_name': 'pg_cnn+', 'solver_cls_name': 'PgCnnSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_cnn+', 'solver_cls_name': 'A2cCnnSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_cnn+', 'solver_cls_name': 'A3cCnnSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_cnn+', 'solver_cls_name': 'PpoCnnSolver', 'rl_solver_cls': PPOSolver},
]
for solver_info in extented_cnn_solvers:
    # make and register the solver class
    solver_name = solver_info['solver_name']
    policy_key = '_'.join(solver_info['solver_name'].split('_')[1:])[:-1]
    policy_builder = build_policy
    obs_as_tensor = TensorConvertor.obs_as_tensor_for_cnn
    instance_env_cls = CnnInstanceEnv
    base_solver_cls = solver_info['rl_solver_cls']
    make_solver_class(solver_name, instance_env_cls, base_solver_cls, policy_builder, obs_as_tensor)
