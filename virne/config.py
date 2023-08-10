# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import json
import time
import pprint
import socket
import networkx as nx

from .utils.class_dict import ClassDict
from .utils.setting import read_setting, write_setting

from typing import Any


class Config(ClassDict):
    """
    Config class for all the settings.
    """
    def __init__(self, **kwargs):
        ### Dataset ###
        path = os.path.dirname(os.path.abspath(__file__))
        self.p_net_setting_path = kwargs.get('p_net_setting_path', os.path.join(path, 'settings/p_net_setting.yaml'))
        self.v_sim_setting_path = kwargs.get('v_sim_setting_path', os.path.join(path, 'settings/v_sim_setting.yaml'))

        ### Environment ###
        self.time_window_size = kwargs.get('time_window_size', 100)
        self.renew_v_net_simulator = False
        self.node_resource_unit_price = 1.
        self.link_resource_unit_price = 1.
        self.revenue_service_time_weight = 0.001
        self.revenue_start_price_weight = 1.
        self.r2c_ratio_threshold = 0.0
        self.vn_size_threshold = 10000
        
        # log & save
        self.if_save_records = True
        self.if_temp_save_records = True
        self.if_save_config = True
        self.summary_dir = 'save/'
        self.save_dir = 'save/'
        self.summary_file_name = 'global_summary.csv'

        ### solver  ###
        self.solver_name = 'nrm_rank'
        self.sub_solver_name = None
        self.pretrained_model_path = ''
        self.pretrained_subsolver_model_path = ''
        # self.solver_name = 'nrm_rank'
        self.verbose = 1                      # Level of showing information 0: no output, 1: output summary, 2: output detailed info
        self.reusable = False                 # Whether or not to allow to deploy several virtual nodes on the same physical node
        ### ranking & mapping ###
        self.node_ranking_method = 'order'    # Method of node ranking: 'order' or 'greedy'
        self.link_ranking_method = 'order'    # Method of link ranking: 'order' or 'greedy'
        self.matching_mathod = 'greedy'       # Method of node matching: 'greedy' or 'l2s2'
        self.shortest_method = 'k_shortest'   # Method of path finding: 'bfs_shortest' or 'k_shortest'
        self.k_shortest = 10                  # Number of shortest paths to be found
        self.allow_revocable = False          # Whether or not to allow to revoke a virtual node
        self.allow_rejection = False          # Whether or not to allow to reject a virtual node

        ### Training ###
        self.num_epochs = 1
        self.seed = None
        self.use_cuda = True
        self.cuda_id = 0
        self.distributed_training = False
        self.num_train_epochs = 100
        self.num_workers = 10
        self.batch_size = 128
        self.target_steps = self.batch_size * 2
        self.repeat_times = 10
        self.save_interval = 10
        self.eval_interval = 10

        ### Neural Network ###
        self.embedding_dim = 128   # Embedding dimension
        self.hidden_dim = 128      # Hidden dimension
        self.num_layers = 1        # Number of GRU stacks' layers
        self.num_gnn_layers = 5    # Number of GNN layers
        self.dropout_prob = 0.0    # Droput rate
        self.batch_norm = False    # Batch normalization
        self.l2reg_rate = 2.5e-4   # L2 regularization rate
        self.lr = 1e-3

        ### Reinforcement Learning ###
        self.rl_gamma = 0.99
        self.explore_rate = 0.9
        self.gae_lambda = 0.98
        self.lr_actor = 1e-3
        self.lr_critic = 1e-3
        self.decode_strategy = 'greedy'
        self.k_searching = 1

        # loss
        self.coef_critic_loss = 0.5
        self.coef_entropy_loss = 0.01
        self.coef_mask_loss = 0.01
        self.reward_weight = 0.1

        self.lr_penalty_params = 1e-3
        self.lr_cost_critic = 1e-3

        # tricks
        self.mask_actions = True
        self.maskable_policy = True
        self.use_negative_sample = True
        self.norm_advantage = True
        self.clip_grad = True
        self.eps_clip = 0.2
        self.max_grad_norm = 0.5
        self.norm_critic_loss = False
        self.use_baseline_solver = False
        self.weight_decay = 0.00001

        ### Safe RL ###
        self.srl_cost_budget = 0.2
        self.srl_alpha = 1.
        self.srl_beta = 0.1

        ### Read settings ###
        self.read_settings(p_net=True, v_sim=True)
        self.create_dirs()
        self.get_run_id()
        check_config(self)

    def get_run_id(self):
        self.run_time = time.strftime('%Y%m%dT%H%M%S')
        self.host_name = socket.gethostname()
        self.run_id = f'{self.host_name}-{self.run_time}'

    def update(self, update_args):
        if not isinstance(update_args, dict):
            update_args = vars(update_args)
        print(f'='*20 + ' Update Default Config ' + '='*20) if self.verbose > 0 else None
        recurisve_update(self, update_args)
        self.target_steps = self.batch_size * 2
        if 'p_net_setting_path' in update_args: self.read_settings(p_net=True, v_sim=False)
        if 'v_sim_setting_path' in update_args: self.read_settings(p_net=False, v_sim=True)
        self.create_dirs()
        check_config(self)
        print(f'='*20 + '=======================' + '='*20) if self.verbose > 0 else None

    def save(self, fname='config.yaml'):
        save_dir = os.path.join(self.save_dir, self.solver_name, self.run_id)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        config_path = os.path.join(save_dir, fname)
        write_setting(vars(self), config_path)
        print(f'Save config in {config_path}')

    @staticmethod
    def load(config_dict):
        return Config.from_dict(config_dict)

    def read_settings(self, p_net=True, v_sim=True):
        # config.general_setting = read_setting(config.general_setting_path)
        # config.nn_setting = read_setting(config.nn_setting_path)
        # config.rl_setting = read_setting(config.rl_setting_path)
        # config.run_setting = read_setting(config.run_setting_path)
        if p_net:
            self.p_net_setting = read_setting(self.p_net_setting_path)
            if 'file_path' in self.p_net_setting['topology'] and os.path.exists(self.p_net_setting['topology']['file_path']):
                G = nx.read_gml(self.p_net_setting['topology']['file_path'], label='id')
                self.p_net_setting['num_nodes'] = G.number_of_nodes()
            self.p_net_setting_num_nodes = self.p_net_setting['num_nodes']
            self.p_net_setting_num_node_attrs = len(self.p_net_setting['node_attrs_setting'])
            self.p_net_setting_num_link_attrs = len(self.p_net_setting['link_attrs_setting'])
            self.p_net_setting_num_node_resource_attrs = len([1 for attr in self.p_net_setting['node_attrs_setting'] if attr['type'] == 'resource'])
            self.p_net_setting_num_link_resource_attrs = len([1 for attr in self.p_net_setting['link_attrs_setting'] if attr['type'] == 'resource'])
            self.p_net_setting_num_node_extrema_attrs = len([1 for attr in self.p_net_setting['node_attrs_setting'] if attr['type'] == 'extrema'])
            self.p_net_setting_num_link_extrema_attrs = len([1 for attr in self.p_net_setting['link_attrs_setting'] if attr['type'] == 'extrema'])
        if v_sim:
            self.v_sim_setting = read_setting(self.v_sim_setting_path)
            self.v_sim_setting_num_node_attrs = len(self.v_sim_setting['node_attrs_setting'])
            self.v_sim_setting_num_link_attrs = len(self.v_sim_setting['link_attrs_setting'])
            self.v_sim_setting_num_node_resource_attrs = len([1 for attr in self.v_sim_setting['node_attrs_setting'] if attr['type'] == 'resource'])
            self.v_sim_setting_num_link_resource_attrs = len([1 for attr in self.v_sim_setting['link_attrs_setting'] if attr['type'] == 'resource'])
            # self.v_sim_setting_aver_lifetime = self.v_sim_setting['aver_lifetime']

    def create_dirs(self):
        self.v_sim_setting['save_dir']
        for dir in [self.save_dir, self.summary_dir, self.v_sim_setting['save_dir'], self.p_net_setting['save_dir']]:
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)

    def __repr__(self):
        return pprint.pformat(self.__dict__)
    
    def __str__(self):
        return pprint.pformat(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()
    
    def values(self):
        return self.__dict__.values()


def update_simulation_setting(
        config: Config, 
        v_sim_setting_num_v_nets: int = None,
        v_sim_setting_v_net_size_low: int = None,
        v_sim_setting_v_net_size_high: int = None,
        v_sim_setting_resource_attrs_high: int = None,
        v_sim_setting_resource_attrs_low: int = None,
        v_sim_setting_aver_lifetime: float = None,
        v_sim_setting_aver_arrival_rate: float = None,
        p_net_setting_num_nodes: int = None,
        p_net_setting_topology_file_path: str = None,
    ) -> None:
    if p_net_setting_topology_file_path is not None:
        assert p_net_setting_num_nodes is None, 'p_net_setting_num_nodes and p_net_setting_topology_file_path cannot be set at the same time'
    if v_sim_setting_num_v_nets is not None:
        config.v_sim_setting['num_v_nets'] = v_sim_setting_num_v_nets
    if v_sim_setting_v_net_size_low is not None:
        config.v_sim_setting['v_net_size']['low'] = v_sim_setting_v_net_size_low
    if v_sim_setting_v_net_size_high is not None:
        config.v_sim_setting['v_net_size']['high'] = v_sim_setting_v_net_size_high
    for i in range(len(config.v_sim_setting['node_attrs_setting'])):
        if config.v_sim_setting['node_attrs_setting'][i]['type'] == 'resource':
            config.v_sim_setting['node_attrs_setting'][i]['high'] = v_sim_setting_resource_attrs_high \
                if v_sim_setting_resource_attrs_high is not None else config.v_sim_setting['node_attrs_setting'][i]['high']
            config.v_sim_setting['node_attrs_setting'][i]['low'] = v_sim_setting_resource_attrs_low \
                if v_sim_setting_resource_attrs_low is not None else config.v_sim_setting['node_attrs_setting'][i]['low']
    for i in range(len(config.v_sim_setting['link_attrs_setting'])):
        if config.v_sim_setting['link_attrs_setting'][i]['type'] == 'resource':
            config.v_sim_setting['link_attrs_setting'][i]['high'] = v_sim_setting_resource_attrs_high \
                if v_sim_setting_resource_attrs_high is not None else config.v_sim_setting['link_attrs_setting'][i]['high']
            config.v_sim_setting['link_attrs_setting'][i]['low'] = v_sim_setting_resource_attrs_low \
                if v_sim_setting_resource_attrs_low is not None else config.v_sim_setting['link_attrs_setting'][i]['low']
    if v_sim_setting_aver_lifetime is not None:
        config.v_sim_setting['lifetime']['scale'] = v_sim_setting_aver_lifetime
    if v_sim_setting_aver_arrival_rate is not None:
        config.v_sim_setting['arrival_rate']['lam'] = v_sim_setting_aver_arrival_rate
    if p_net_setting_num_nodes is not None:
        config.p_net_setting['num_nodes'] = p_net_setting_num_nodes
    if p_net_setting_topology_file_path is not None:
        config.p_net_setting['topology']['file_path'] = p_net_setting_topology_file_path
        G = nx.read_gml(config.p_net_setting['topology']['file_path'], label='id')
        config.p_net_setting['num_nodes'] = G.number_of_nodes()
        config.p_net_setting['num_links'] = G.number_of_edges()

def recurisve_update(config: 'Config', update_args: dict) -> None:
    """Recursively update args.

    Args:
        update_args (Dict[str, Any]): args to be updated.
    """
    for key, value in update_args.items():
        if key in config.keys():
            if isinstance(value, dict):
                recurisve_update(config[key], value)
            else:
                if config[key] != value:
                    config[key] = value
                    print(f'Update {key} with {value}') if config.verbose > 0 else None
        else:
            config[key] = value
            print(f'Add {key} with {value}') if config.verbose > 0 else None


def check_config(config: Config) -> None:
    """Check all configs.

    This function is used to check the configs.

    Args:
        configs (dict): configs to be checked.
        algo_type (str): algorithm type.
    """
    assert config.reusable == False, 'Unsupported currently!'
    if config.target_steps != -1:
        assert config.target_steps % config.batch_size == 0, 'A should be greater than b!'

def show_config(config):
    pprint.pprint(vars(config))

def save_config(config, fname='config.yaml'):
    save_dir = os.path.join(config.save_dir, config.solver_name, config.run_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    config_path = os.path.join(save_dir, fname)
    write_setting(vars(config), config_path)
    print(f'Save config in {config_path}')

def load_config(fpath=None):
    try:
        config = read_setting(fpath)
        print(f'Load config from {fpath}')
    except:
        print(f'No config file found in {fpath}')
    return config

def set_sim_info_to_object(config_dict: dict, obj):
    if not isinstance(config_dict, dict):
        config_dict = vars(config_dict)
    for key in [
        'p_net_setting_num_nodes', 
        'p_net_setting_num_node_attrs', 
        'p_net_setting_num_link_attrs', 
        'p_net_setting_num_node_resource_attrs', 
        'p_net_setting_num_link_resource_attrs', 
        'p_net_setting_num_node_extrema_attrs'
        ]:
        setattr(obj, key, config_dict[key]) if not hasattr(obj, key) else None
    for key in [
        'v_sim_setting_num_node_attrs',
        'v_sim_setting_num_link_attrs',
        'v_sim_setting_num_node_resource_attrs',
        'v_sim_setting_num_link_resource_attrs'
        ]:
        setattr(obj, key, config_dict[key]) if not hasattr(obj, key) else None