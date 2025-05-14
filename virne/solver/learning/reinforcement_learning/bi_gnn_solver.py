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
from virne.solver.learning.rl_policy import CnnActorCritic
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry, PNetVNodeFeatureConstructor
from virne.solver.learning.rl_core.reward_calculator import RewardCalculatorRegistry, GradualIntermediateRewardCalculator
from virne.solver.learning.reinforcement_learning.solver_maker import make_solver_class

obs_as_tensor = TensorConvertor.obs_as_tensor_for_bi_gnn


class BiGnnInstanceEnv(JointPRStepInstanceRLEnv):
    """
    RL environment for Bi-GNN-based solvers.
    """
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(BiGnnInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'
        self.feature_constructor = FeatureConstructorRegistry.get(config.rl.feature_constructor.name)(
            self.node_attr_benchmarks or {},
            self.link_attr_benchmarks or {},
            self.link_sum_attr_benchmarks or {},
            self.config
        )



class BiGcnInstanceEnv(JointPRStepInstanceRLEnv):
    """
    RL environment for Bi-GCN-based solvers.
    """
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super(BiGcnInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'
            config.rl.feature_constructor.if_use_degree_metric = False
            config.rl.feature_constructor.if_use_more_topological_metrics = False
            config.rl.feature_constructor.if_use_aggregated_link_attrs = True
            config.rl.reward_calculator.name = 'gradual_intermediate'
        self.feature_constructor = FeatureConstructorRegistry.get(config.rl.feature_constructor.name)(
            self.node_attr_benchmarks or {},
            self.link_attr_benchmarks or {},
            self.link_sum_attr_benchmarks or {},
            self.config
        )
        self.reward_calculator = RewardCalculatorRegistry.get(config.rl.reward_calculator.name)(self.config)


# @SolverRegistry.register(solver_name='acn_bi_gcn', solver_type='r_learning')
@SolverRegistry.register(solver_name='acn_bi_gcn', solver_type='r_learning')
class BiGcnSolver(InstanceAgent, PGSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, BiGcnInstanceEnv)
        PGSolver.__init__(self, controller, recorder, counter, logger, config, PolicyBuilder.build_bi_gcn_policy, obs_as_tensor, **kwargs)


# Policy builder registry for extensibility
POLICY_BUILDERS = {
    'bi_gcn': PolicyBuilder.build_bi_gcn_policy,
    'bi_gat': PolicyBuilder.build_bi_gat_policy,
    'bi_deep_edge_gat': PolicyBuilder.build_bi_deep_edge_gat_policy,
}

extented_gnn_mlp_solvers = [
    {'solver_name': 'pg_bi_gcn+', 'policy_key': 'bi_gcn', 'solver_cls_name': 'PgGcnMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_bi_gcn+', 'policy_key': 'bi_gcn', 'solver_cls_name': 'A2cGcnMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_bi_gcn+', 'policy_key': 'bi_gcn', 'solver_cls_name': 'BiGcnSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_bi_gcn+', 'policy_key': 'bi_gcn', 'solver_cls_name': 'PpoGcnMlpSolver', 'rl_solver_cls': PPOSolver},

    {'solver_name': 'pg_bi_gat+', 'policy_key': 'bi_gat', 'solver_cls_name': 'PgGatMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_bi_gat+', 'policy_key': 'bi_gat', 'solver_cls_name': 'A2cGatMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_bi_gat+', 'policy_key': 'bi_gat', 'solver_cls_name': 'A3cGatMlpSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_bi_gat+', 'policy_key': 'bi_gat', 'solver_cls_name': 'PpoGatMlpSolver', 'rl_solver_cls': PPOSolver},

    {'solver_name': 'pg_bi_deep_edge_gat+', 'policy_key': 'bi_deep_edge_gat', 'solver_cls_name': 'PgEdgeGatMlpSolver', 'rl_solver_cls': PGSolver},
    {'solver_name': 'a2c_bi_deep_edge_gat+', 'policy_key': 'bi_deep_edge_gat', 'solver_cls_name': 'A2cEdgeGatMlpSolver', 'rl_solver_cls': A2CSolver},
    {'solver_name': 'a3c_bi_deep_edge_gat+', 'policy_key': 'bi_deep_edge_gat', 'solver_cls_name': 'A3cEdgeGatMlpSolver', 'rl_solver_cls': A3CSolver},
    {'solver_name': 'ppo_bi_deep_edge_gat+', 'policy_key': 'bi_deep_edge_gat', 'solver_cls_name': 'PpoEdgeGatMlpSolver', 'rl_solver_cls': PPOSolver},
]

for solver_info in extented_gnn_mlp_solvers:
    # make and register the solver class
    solver_name = solver_info['solver_name']
    policy_key = '_'.join(solver_info['solver_name'].split('_')[1:])[:-1]
    policy_builder = POLICY_BUILDERS[policy_key]
    obs_as_tensor = obs_as_tensor
    instance_env_cls = BiGnnInstanceEnv
    base_solver_cls = solver_info['rl_solver_cls']

    make_solver_class(solver_name, instance_env_cls, base_solver_cls, policy_builder, obs_as_tensor)

