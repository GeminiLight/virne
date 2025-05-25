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
from virne.solver.learning.rl_policy import CnnActorCritic
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder
from virne.solver.learning.reinforcement_learning.solver_maker import make_solver_class

obs_as_tensor = TensorConvertor.obs_as_tensor_for_dual_gnn


class DualGnnInstanceEnv(JointPRStepInstanceRLEnv):
    """
    RL environment for Dual-GNN-based solvers.
    """
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'
        super(DualGnnInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


class DualGcnInstanceEnv(JointPRStepInstanceRLEnv):
    """
    RL environment for Dual-GCN-based solvers.
    """
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'
            config.rl.feature_constructor.if_use_degree_metric = False
            config.rl.feature_constructor.if_use_more_topological_metrics = False
            config.rl.feature_constructor.if_use_aggregated_link_attrs = True
            config.rl.reward_calculator.name = 'gradual_intermediate'
        super(DualGcnInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


@SolverRegistry.register(solver_name='a2c_dual_gcn', solver_type='r_learning')
class A2CDualGcnSolver(InstanceAgent, A2CSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, DualGcnInstanceEnv)
        A2CSolver.__init__(self, controller, recorder, counter, logger, config, PolicyBuilder.build_dual_gcn_policy, obs_as_tensor, **kwargs)

@SolverRegistry.register(solver_name='ppo_dual_gcn', solver_type='r_learning')
class PpoDualGcnSolver(InstanceAgent, PPOSolver):
    """A PPO solver for Dual-GCN-based solvers.

    References:
    
    
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, DualGcnInstanceEnv)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, PolicyBuilder.build_dual_gcn_policy, obs_as_tensor, **kwargs)



POLICY_BUILDERS = {
    'dual_gcn': PolicyBuilder.build_dual_gcn_policy,
    'dual_gat': PolicyBuilder.build_dual_gat_policy,
    'dual_deep_edge_gat': PolicyBuilder.build_dual_deep_edge_gat_policy,
}

extented_gnn_mlp_solvers = [
    {'solver_name': 'pg_dual_gcn+', 'policy_key': 'dual_gcn', 'solver_cls_name': 'PgGcnMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_dual_gcn+', 'policy_key': 'dual_gcn', 'solver_cls_name': 'A2cGcnMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_dual_gcn+', 'policy_key': 'dual_gcn', 'solver_cls_name': 'DualGcnSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_dual_gcn+', 'policy_key': 'dual_gcn', 'solver_cls_name': 'PpoGcnMlpSolver', 'rl_solver_cls': PPOSolver},

    {'solver_name': 'pg_dual_gat+', 'policy_key': 'dual_gat', 'solver_cls_name': 'PgGatMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_dual_gat+', 'policy_key': 'dual_gat', 'solver_cls_name': 'A2cGatMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_dual_gat+', 'policy_key': 'dual_gat', 'solver_cls_name': 'A3cGatMlpSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_dual_gat+', 'policy_key': 'dual_gat', 'solver_cls_name': 'PpoGatMlpSolver', 'rl_solver_cls': PPOSolver},

    {'solver_name': 'pg_dual_deep_edge_gat+', 'policy_key': 'dual_deep_edge_gat', 'solver_cls_name': 'PgEdgeGatMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_dual_deep_edge_gat+', 'policy_key': 'dual_deep_edge_gat', 'solver_cls_name': 'A2cEdgeGatMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_dual_deep_edge_gat+', 'policy_key': 'dual_deep_edge_gat', 'solver_cls_name': 'A3cEdgeGatMlpSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_dual_deep_edge_gat+', 'policy_key': 'dual_deep_edge_gat', 'solver_cls_name': 'PpoEdgeGatMlpSolver', 'rl_solver_cls': PPOSolver},
]

for solver_info in extented_gnn_mlp_solvers:
    # make and register the solver class
    solver_name = solver_info['solver_name']
    policy_key = '_'.join(solver_info['solver_name'].split('_')[1:])[:-1]
    policy_builder = POLICY_BUILDERS[policy_key]
    obs_as_tensor = obs_as_tensor
    instance_env_cls = DualGnnInstanceEnv
    base_solver_cls = solver_info['rl_solver_cls']

    make_solver_class(solver_name, instance_env_cls, base_solver_cls, policy_builder, obs_as_tensor)

