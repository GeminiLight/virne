# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from typing import Any, Dict, Tuple, List, Union, Optional, Type, Callable

import os
import numpy as np
import torch
import torch.nn as nn
from omegaconf import open_dict

from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger

from virne.solver import SolverRegistry
from virne.solver.learning.rl_policy import CnnActorCritic
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder, OptimizerBuilder
from virne.solver.learning.reinforcement_learning.solver_maker import make_solver_class


from .net import ActorCritic
from virne.solver.learning.rl_core import RLSolver, PPOSolver, A2CSolver, InstanceAgent, A3CSolver, NeuralLagrangianPPOSolver, SafeInstanceAgent
from virne.solver.base_solver import SolverRegistry






obs_as_tensor = TensorConvertor.obs_as_tensor_for_dual_gnn
# make_policy = PolicyBuilder.build_dual_gcn_policy

class BiGnnInstanceEnv(JointPRStepInstanceRLEnv):
    """
    RL environment for BiGNN-based solvers.
    """
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'
            config.rl.reward_calculator.name = 'vanilla'
        super(BiGnnInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)



@SolverRegistry.register(solver_name='safe_ppo_gcn', solver_type='r_learning')
class SafePpoGcnSolver(SafeInstanceAgent, NeuralLagrangianPPOSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        SafeInstanceAgent.__init__(self, BiGnnInstanceEnv)
        NeuralLagrangianPPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.compute_cost_method = kwargs.get('compute_cost_method', 'reachability')
        print(f'Compute cost method: {self.compute_cost_method}')


def make_policy(agent) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Build a Bi-GCN-based actor-critic policy and its optimizer.
    """
    # feature_dim_config = PolicyBuilder.get_feature_dim_config(agent.config)
    policy = ActorCritic(
        **PolicyBuilder.get_feature_dim_config(agent.config),
        **PolicyBuilder.get_general_nn_config(agent.config),
    ).to(agent.device)

    optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
    return policy, optimizer
