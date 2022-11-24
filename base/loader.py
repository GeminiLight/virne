from .enironment import SolutionStepEnvironment
from solver.exact.mip_vne import *
from solver.heuristic.node_rank import *
from solver.heuristic.joint_pr import *
from solver.heuristic.bfs_trials import *


def load_simulator(solver_name):
    # exact
    if solver_name == 'mip_vne':
        Env, Solver = SolutionStepEnvironment, MIPSolver
    elif solver_name == 'r_vne':
        from solver.exact.r_vne import RandomizedRoundingSolver
        Env, Solver = SolutionStepEnvironment, RandomizedRoundingSolver
    elif solver_name == 'd_vne':
        from solver.exact.d_vne import DeterministicRoundingSolver
        Env, Solver = SolutionStepEnvironment, DeterministicRoundingSolver
    # rank
    elif solver_name == 'random_rank':
        Env, Solver = SolutionStepEnvironment, RandomRankSolver
    elif solver_name == 'order_rank':
        Env, Solver = SolutionStepEnvironment, OrderRankSolver
    elif solver_name == 'rw_rank':
        Env, Solver = SolutionStepEnvironment, RandomWalkRankSolver
    elif solver_name == 'grc_rank':
        Env, Solver = SolutionStepEnvironment, GRCRankSolver
    elif solver_name == 'ffd_rank':
        Env, Solver = SolutionStepEnvironment, FFDRankSolver
    elif solver_name == 'nrm_rank':
        Env, Solver = SolutionStepEnvironment, NRMRankSolver
    elif solver_name == 'pl_rank':
        Env, Solver = SolutionStepEnvironment, PLRankSolver
    # joint_pr
    elif solver_name == 'random_joint_pr':
        Env, Solver = SolutionStepEnvironment, RandomJointPRSolver
    elif solver_name == 'order_joint_pr':
        Env, Solver = SolutionStepEnvironment, OrderJointPRSolver
    elif solver_name == 'ffd_joint_pr':
        Env, Solver = SolutionStepEnvironment, FFDJointPRSolver
    # rank_bfs
    elif solver_name == 'random_rank_bfs':
        Env, Solver = SolutionStepEnvironment, RandomRankBFSSolver
    elif solver_name == 'rw_rank_bfs':
        Env, Solver = SolutionStepEnvironment, RandomWalkRankBFSSolver
    elif solver_name == 'order_rank_bfs':
        Env, Solver = SolutionStepEnvironment, OrderRankBFSSolver
    # meta-heuristic
    elif solver_name == 'pso_vne':
        from solver.meta_heuristic.particle_swarm_optimization_solver import ParticleSwarmOptimizationSolver
        Env, Solver = SolutionStepEnvironment, ParticleSwarmOptimizationSolver
    elif solver_name == 'aco_vne':
        from solver.meta_heuristic.ant_colony_optimization_solver import AntColonyOptimizationSolver
        Env, Solver = SolutionStepEnvironment, AntColonyOptimizationSolver
    elif solver_name == 'sa_vne':
        from solver.meta_heuristic.simulated_annealing_solver import SimulatedAnnealingSolver
        Env, Solver = SolutionStepEnvironment, SimulatedAnnealingSolver
    elif solver_name == 'ga_vne':
        from solver.meta_heuristic.genetic_algorithm_solver import GeneticAlgorithmSolver
        Env, Solver = SolutionStepEnvironment, GeneticAlgorithmSolver
    # ml
    elif solver_name == 'neuro_vne':
        from solver.learning.neuro_vne.neuro_vne import NeuroSolver
        Env, Solver = SolutionStepEnvironment, NeuroSolver
    # rl
    elif solver_name == 'mcts_vne':
        from solver.learning.mcts_vne import MCTSSolver
        Env, Solver = SolutionStepEnvironment, MCTSSolver
    elif solver_name == 'pg_mlp':
        from solver.learning.pg_mlp import PGMLPSolver
        Env, Solver = SolutionStepEnvironment, PGMLPSolver
    elif solver_name == 'pg_mlp2':
        from solver.learning.pg_mlp2 import PGMLP2Solver
        Env, Solver = SolutionStepEnvironment, PGMLP2Solver
    elif solver_name == 'pg_cnn':
        from solver.learning.pg_cnn import PgCnnSolver
        Env, Solver = SolutionStepEnvironment, PgCnnSolver
    elif solver_name == 'pg_cnn2':
        from solver.learning.pg_cnn2 import PgCnn2Solver
        Env, Solver = SolutionStepEnvironment, PgCnn2Solver
    elif solver_name == 'pg_seq2seq':
        from solver.learning.pg_seq2seq import PGSeq2SeqSolver
        Env, Solver = SolutionStepEnvironment, PGSeq2SeqSolver
    else:
        raise ValueError('The solver is not yet supported; \n Please attempt to select another one.', solver_name)
    return Env, Solver
