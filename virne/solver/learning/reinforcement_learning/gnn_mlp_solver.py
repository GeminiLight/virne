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
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry, PNetVNodeFeatureConstructor
from virne.solver.learning.rl_core.reward_calculator import RewardCalculatorRegistry, GradualIntermediateRewardCalculator
from virne.solver.learning.reinforcement_learning.solver_maker import make_solver_class

obs_as_tensor = TensorConvertor.obs_as_tensor_for_gnn_mlp


class GnnMlpInstanceEnv(JointPRStepInstanceRLEnv):
    """
    RL environment for GNN-MLP-based solvers.
    """
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
        super(GnnMlpInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)

class A3cGcnMlpInstanceEnv(JointPRStepInstanceRLEnv):
    """
    RL environment for GNN-MLP-based solvers.
    """
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config: DictConfig, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_node'
            config.rl.feature_constructor.if_use_degree_metric = False
            config.rl.feature_constructor.if_use_more_topological_metrics = False
            config.rl.feature_constructor.if_use_aggregated_link_attrs = True
            config.rl.feature_constructor.if_use_node_status_flags = True
        super(A3cGcnMlpInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


@SolverRegistry.register(solver_name='a3n_gcn_mlp', solver_type='r_learning')
class A3cGcnSolver(InstanceAgent, PGSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, A3cGcnMlpInstanceEnv)
        PGSolver.__init__(self, controller, recorder, counter, logger, config, PolicyBuilder.build_gcn_mlp_policy, obs_as_tensor, **kwargs)


# Policy builder registry for extensibility
POLICY_BUILDERS = {
    'gcn_mlp': PolicyBuilder.build_gcn_mlp_policy,
    'gat_mlp': PolicyBuilder.build_gat_mlp_policy,
    'deep_edge_gat_mlp': PolicyBuilder.build_deep_edge_gat_mlp_policy,
}

extented_gnn_mlp_solvers = [
    {'solver_name': 'pg_gcn_mlp+', 'policy_key': 'gcn_mlp', 'solver_cls_name': 'PgGcnMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_gcn_mlp+', 'policy_key': 'gcn_mlp', 'solver_cls_name': 'A2cGcnMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_gcn_mlp+', 'policy_key': 'gcn_mlp', 'solver_cls_name': 'A3cGcnMlpSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_gcn_mlp+', 'policy_key': 'gcn_mlp', 'solver_cls_name': 'PpoGcnMlpSolver', 'rl_solver_cls': PPOSolver},

    {'solver_name': 'pg_gat_mlp+', 'policy_key': 'gat_mlp', 'solver_cls_name': 'PgGatMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_gat_mlp+', 'policy_key': 'gat_mlp', 'solver_cls_name': 'A2cGatMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_gat_mlp+', 'policy_key': 'gat_mlp', 'solver_cls_name': 'A3cGatMlpSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_gat_mlp+', 'policy_key': 'gat_mlp', 'solver_cls_name': 'PpoGatMlpSolver', 'rl_solver_cls': PPOSolver},

    {'solver_name': 'pg_deep_edge_gat_mlp+', 'policy_key': 'deep_edge_gat_mlp', 'solver_cls_name': 'PgEdgeGatMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_deep_edge_gat_mlp+', 'policy_key': 'deep_edge_gat_mlp', 'solver_cls_name': 'A2cEdgeGatMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_deep_edge_gat_mlp+', 'policy_key': 'deep_edge_gat_mlp', 'solver_cls_name': 'A3cEdgeGatMlpSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_deep_edge_gat_mlp+', 'policy_key': 'deep_edge_gat_mlp', 'solver_cls_name': 'PpoEdgeGatMlpSolver', 'rl_solver_cls': PPOSolver},
]

for solver_info in extented_gnn_mlp_solvers:
    # make and register the solver class
    solver_name = solver_info['solver_name']
    policy_key = '_'.join(solver_info['solver_name'].split('_')[1:])[:-1]
    policy_builder = POLICY_BUILDERS[policy_key]
    obs_as_tensor = TensorConvertor.obs_as_tensor_for_gnn_mlp
    instance_env_cls = GnnMlpInstanceEnv
    base_solver_cls = solver_info['rl_solver_cls']

    make_solver_class(solver_name, instance_env_cls, base_solver_cls, policy_builder, obs_as_tensor)
