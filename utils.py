import os
import json

from base import SolutionStepEnvironment
from solver.heuristics.node_rank import *
from solver.heuristics.joint_pr import *
from solver.heuristics.bfs_trials import *


def read_json(fpath):
    with open(fpath, 'r') as f:
        attrs_dict = json.load(f)
    return attrs_dict

def write_json(dict_data, fpath):
    with open(fpath, 'w+') as f:
        json.dump(dict_data, f)

def load_solver(solver_name):
    # rank
    if solver_name == 'random_rank':
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
    # exact
    elif solver_name == 'd_vne':
        from solver.exact.d_vne import DeterministicRoundingSolver
        Env, Solver = SolutionStepEnvironment, DeterministicRoundingSolver
    # meta-heuristic
    elif solver_name == 'pso_vne':
        from solver.meta_heuristics.pso import PSOSolver
        Env, Solver = SolutionStepEnvironment, PSOSolver
    elif solver_name == 'aco_vne':
        from solver.meta_heuristics.aco import ACOSolver
        Env, Solver = SolutionStepEnvironment, ACOSolver
    # ml
    elif solver_name == 'mcts_vne':
        from solver.learning.mcts_vne import MCTSSolver
        Env, Solver = SolutionStepEnvironment, MCTSSolver
    elif solver_name == 'gae_vne':
        from solver.learning.gae_vne import GAESolver
        Env, Solver = SolutionStepEnvironment, GAESolver
    elif solver_name == 'neuro_vne':
        from solver.learning.neuro_vne.neuro_vne import NeuroSolver
        Env, Solver = SolutionStepEnvironment, NeuroSolver
    # rl
    elif solver_name == 'pg_cnn':
        from solver.learning.pg_cnn import PGCNNSolver
        Env, Solver = SolutionStepEnvironment, PGCNNSolver
    elif solver_name == 'pg_cnn2':
        from solver.learning.pg_cnn2 import PGCNN2Solver
        Env, Solver = SolutionStepEnvironment, PGCNN2Solver
    elif solver_name == 'pg_seq2seq':
        from solver.learning.pg_seq2seq import PGSeq2SeqSolver
        Env, Solver = SolutionStepEnvironment, PGSeq2SeqSolver
    elif solver_name == 'ppo_gnn':
        from solver.learning.ppo_gnn_old import PPOGNNSolver
        Env, Solver = SolutionStepEnvironment, PPOGNNSolver
    elif solver_name == 'a2c_gat':
        from solver.learning.a2c_gat import A2CGATSolver
        Env, Solver = SolutionStepEnvironment, A2CGATSolver
    elif solver_name == 'ppo_gat':
        from solver.learning.ppo_gat import PPOGATSolver
        Env, Solver = SolutionStepEnvironment, PPOGATSolver
    elif solver_name == 'ppo_gnn2':
        from solver.learning.ppo_gnn2 import PPOGNNSolver
        Env, Solver = SolutionStepEnvironment, PPOGNNSolver
    elif solver_name == 'ns_gnn':
        from solver.learning.ns_gnn.ns_gnn_solver import NSGNNSolver
        Env, Solver = SolutionStepEnvironment, NSGNNSolver
    else:
        raise ValueError('The solverrithm is not yet supported; \n Please attempt to select another one.', solver_name)
    return Env, Solver

def get_pn_dataset_dir_from_setting(pn_setting):
    pn_dataset_dir = pn_setting.get('save_dir')
    n_attrs = [n_attr['name'] for n_attr in pn_setting['node_attrs_setting']]
    e_attrs = [e_attr['name'] for e_attr in pn_setting['edge_attrs_setting']]

    pn_dataset_middir = f"{pn_setting['num_nodes']}-{pn_setting['topology']['type']}-{pn_setting['topology']['wm_alpha']}-{pn_setting['topology']['wm_beta']}-" +\
                        f"{n_attrs}-[{pn_setting['node_attrs_setting'][0]['low']}-{pn_setting['node_attrs_setting'][0]['high']}]-" + \
                        f"{e_attrs}-[{pn_setting['edge_attrs_setting'][0]['low']}-{pn_setting['edge_attrs_setting'][0]['high']}]"        
    pn_dataset_dir = os.path.join(pn_dataset_dir, pn_dataset_middir)
    return pn_dataset_dir

def get_vns_dataset_dir_from_setting(vns_setting):
    vns_dataset_dir = vns_setting.get('save_dir')
    n_attrs = [n_attr['name'] for n_attr in vns_setting['node_attrs_setting']]
    e_attrs = [e_attr['name'] for e_attr in vns_setting['edge_attrs_setting']]
    
    vns_dataset_middir = f"{vns_setting['num_vns']}-[{vns_setting['vn_size']['low']}-{vns_setting['vn_size']['high']}]-" + \
                        f"{vns_setting['topology']['type']}-{vns_setting['lifetime']['scale']}-{vns_setting['arrival_rate']['lam']}-" + \
                        f"{n_attrs}-[{vns_setting['node_attrs_setting'][0]['low']}-{vns_setting['node_attrs_setting'][0]['high']}]-" + \
                        f"{e_attrs}-[{vns_setting['edge_attrs_setting'][0]['low']}-{vns_setting['edge_attrs_setting'][0]['high']}]"
    vn_dataset_dir = os.path.join(vns_dataset_dir, vns_dataset_middir)

    return vn_dataset_dir

def generate_file_name(config, epoch_id=0, extra_items=[], **kwargs):
    if not isinstance(config, dict): config = vars(config)
    items = extra_items + ['pn_num_nodes', 'reusable']

    file_name_1 = f"{config['solver_name']}-records-{epoch_id}-"
    # file_name_2 = '-'.join([f'{k}={config[k]}' for k in items])
    file_name_3 = '-'.join([f'{k}={v}' for k, v in kwargs.items()])
    file_name = file_name_1 + file_name_3 + '.csv'
    return file_name

def delete_temp_files(file_path):
    del_list = os.listdir(file_path)
    for f in del_list:
        file_path = os.path.join(del_list, f)
        if os.path.isfile(file_path) and 'temp' in file_path:
            os.remove(file_path)