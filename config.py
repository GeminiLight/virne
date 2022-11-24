from cProfile import label
import os
import time
import pprint
import socket
import argparse
import networkx as nx

from utils import read_setting, write_setting


parser = argparse.ArgumentParser(description='configuration file')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return False if v == 0 else True
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

### Dataset ###
data_arg = parser.add_argument_group('data')

setting_arg = parser.add_argument_group('settings')
setting_arg.add_argument('--p_net_setting_path', type=str, default='settings/p_net_setting.yaml', help='File path of physical network settings')
setting_arg.add_argument('--v_sim_setting_path', type=str, default='settings/v_sim_setting.yaml', help='File path of virtual network settings')
setting_arg.add_argument('--nn_setting_path', type=str, default='settings/neural_network.yaml', help='File path of neural network settings')
setting_arg.add_argument('--rl_setting_path', type=str, default='settings/reinforcement_learning.yaml', help='File path of advanced reinforcement learning settings')
setting_arg.add_argument('--run_setting_path', type=str, default='settings/running_and_training.yaml', help='File path of advanced running and training settings')

setting_arg.add_argument('--if_adjust_v_sim_setting', type=str2bool, default=False, help='Whether to deirectly adjust VN settings in parser')
setting_arg.add_argument('--v_sim_setting_max_length', type=int, default=10, help='')
setting_arg.add_argument('--v_sim_setting_aver_arrival_rate', type=float, default=0.10, help='')
setting_arg.add_argument('--v_sim_setting_aver_lifetime', type=int, default=1000, help='')
setting_arg.add_argument('--v_sim_setting_high_request', type=int, default=30, help='')
setting_arg.add_argument('--v_sim_setting_low_request', type=int, default=0, help='')
setting_arg.add_argument('--v_sim_setting_num_v_nets', type=int, default=1000, help='')

### Environment ###
env_arg = parser.add_argument_group('env')
env_arg.add_argument('--summary_dir', type=str, default='save/', help='File Directory to save summary and records')
env_arg.add_argument('--summary_file_name', type=str, default='global_summary.csv', help='Summary file name')
env_arg.add_argument('--if_save_records', type=str2bool, default=True, help='Whether to save records')
env_arg.add_argument('--if_temp_save_records', type=str2bool, default=True, help='Whether to temporarily save records')
env_arg.add_argument('--node_resource_unit_price', type=float, default=1., help='') 
env_arg.add_argument('--link_resource_unit_price', type=float, default=1., help='') 
env_arg.add_argument('--time_window_size', type=int, default=100., help='The time window for batch VNE, only working for BatchScenario') 

### solver  ###
solver_arg = parser.add_argument_group('solver')
solver_arg.add_argument('--solver_name', type=str, default='pg_cnn2', help='Name of the  selected to run')
solver_arg.add_argument('--verbose', type=str2bool, default=1, help='Level of showing information')
solver_arg.add_argument('--reusable', type=str2bool, default=False, help='Whether or not to allow to deploy several VN nodes on the same VNF')

solver_arg.add_argument('--matching_mathod', type=str, default="greedy", help='Node placing approches for node mapping: [greedy, l2s2 (Large2LargeSmall2Small)]') 
solver_arg.add_argument('--shortest_method', type=str, default="bfs_shortest", help='Path finding approches for link mapping: [mcf (Multi-commodity Flow), first_shortest, k_shortest, all_shortest, bfs_shortest, available_shortest]') 
solver_arg.add_argument('--node_ranking_method', type=str, default="order", help='Pre-rank nodes: [order, ffd, grc, nrm, rw]') 
solver_arg.add_argument('--link_ranking_method', type=str, default="order", help='Pre-rank links: [order, ffd]') 
solver_arg.add_argument('--k_shortest', type=int, default=10, help="k param for k_shortest") 
solver_arg.add_argument('--allow_revocable', type=str2bool, default=False, help='')
solver_arg.add_argument('--allow_rejection', type=str2bool, default=False, help='')

### Neural Network ###


### Reinforcement Learning ###
rl_arg = parser.add_argument_group('reinforcement learning')
# rl
rl_arg.add_argument('--rl_gamma', type=float, default=0.95, help='Cumulative reward discount rate')
rl_arg.add_argument('--gae_lambda', type=float, default=0.98, help='')
rl_arg.add_argument('--explore_rate', type=float, default=0.90, help='Epsilon-greedy explore rate')
rl_arg.add_argument('--lr', type=float, default=1e-3, help='General Learning rate')
rl_arg.add_argument('--lr_actor', type=float, default=1e-3, help='Actor learning rate')
rl_arg.add_argument('--lr_critic', type=float, default=1e-3, help='Critic learning rate')
rl_arg.add_argument('--k_searching', type=int, default=3, help='Beam search width or Sample workers number') 
rl_arg.add_argument('--decode_strategy', type=str, default='greedy', help='Solution Decoding Strategy: [greedy (Greedy search) | sample (Sample search)/ beam (Beam search)]') 

### Training ###
train_arg = parser.add_argument_group('train')
train_arg.add_argument('--use_cuda', type=str2bool, default=True, help='Whether to sse GPU to accelerate the training process')
train_arg.add_argument('--cuda_id', type=int, default=0, help='CUDA device id')
train_arg.add_argument('--allow_parallel', type=str2bool, default=False, help='Whether to use mutiple GPUs')
train_arg.add_argument('--batch_size', type=int, default=256, help='Batch size of training')
train_arg.add_argument('--num_train_epochs', type=int, default=100, help='Number of training epochs')
train_arg.add_argument('--num_workers', type=int, default=10, help='Number of sub workers who collect experience asynchronously')
train_arg.add_argument('--target_steps', type=int, default=-1, help='Target steps for collecting experience')
train_arg.add_argument('--repeat_times', type=int, default=10, help='')
train_arg.add_argument('--use_negative_sample', type=str2bool, default=True, help='Whether to allow use failed sample to train')
train_arg.add_argument('--pretrained_model_path', type=str, default='', help='Path of pretrained model')

### Run ###
run_arg = parser.add_argument_group('run')
run_arg.add_argument('--seed', type=int, default=None, help='Random seed')
run_arg.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
run_arg.add_argument('--if_save_config', type=str2bool, default=True, help='Whether to save the config file')
# run_arg.add_argument('--renew_v_net_simulator', type=str2bool, default=False, help='')
# run_arg.add_argument('--start_epoch', type=int, default=0, help='Start from epochi')
# run_arg.add_argument('--only_test', type=str2bool, default=False, help='Only test without training')
# run_arg.add_argument('--save_dir', type=str, default='save', help='Save directory for models and trainning logs')
# run_arg.add_argument('--log_dir', type=str, default='log', help='Log directory for models and trainning logs')
# run_arg.add_argument('--open_tb', type=str2bool, default=True, help='')
# run_arg.add_argument('--log_interval', type=int, default=1, help='')
# run_arg.add_argument('--save_interval', type=int, default=10, help='')
# run_arg.add_argument('--eval_interval', type=int, default=10, help='')
# run_arg.add_argument('--total_steps', type=int, default=5000000, help='')
# run_arg.add_argument('--sub_solver', type=str, default='nrm_rank', help='')
# run_arg.add_argument('--reward_weight', type=float, default=0.1, help='')
# run_arg.add_argument('--save_model', type=str2bool, default=False, help='Save model')
# run_arg.add_argument('--load_model', type=str2bool, default=False, help='Load model')
#run_arg.add_argument('--lr_decay_step', type=int, default=5000, help='Lr1 decay step')
#run_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='Lr1 decay rate')

### Misc ###
misc_arg = parser.add_argument_group('misc')


def get_config(args=None, adjust_p_net_setting={}, adjust_v_sim_setting={}):
    config = parser.parse_args(args)
    neural_network_settings = read_setting(config.nn_setting_path)
    rl_settings = read_setting(config.rl_setting_path)
    run_settings = read_setting(config.run_setting_path)
    for k, v in neural_network_settings.items(): setattr(config, k, v)
    for k, v in rl_settings.items(): setattr(config, k, v)
    for k, v in run_settings.items(): setattr(config, k, v)

    if config.reusable:
        print('*' * 40)
        print(' ' * 5 + '!!! Physical Node Hosts is Reusable !!!')
        print('*' * 40)

    # make dir
    for dir in [config.save_dir, config.summary_dir, 'dataset', 'dataset/p_net', 'dataset/v_nets']:
        if not os.path.exists(dir):
            os.makedirs(dir)
            
    # read p_net and v_nets setting
    config.p_net_setting = read_setting(config.p_net_setting_path)
    if 'file_path' in config.p_net_setting['topology'] and os.path.exists(config.p_net_setting['topology']['file_path']):
        G = nx.read_gml(config.p_net_setting['topology']['file_path'], label='id')
        config.p_net_setting['num_nodes'] = G.number_of_nodes()
    config.v_sim_setting = read_setting(config.v_sim_setting_path)

    config.p_net_setting.update(adjust_p_net_setting)
    config.v_sim_setting.update(adjust_v_sim_setting)

    if config.if_adjust_v_sim_setting:
        config.v_sim_setting['max_length'] = config.v_sim_setting_max_length
        config.v_sim_setting['aver_arrival_rate'] = config.v_sim_setting_aver_arrival_rate
        config.v_sim_setting['aver_lifetime'] = config.v_sim_setting_aver_lifetime
        for n_attr in config.v_sim_setting['node_attrs_setting']: n_attr['high'] = config.v_sim_setting_high_request
        for l_attr in config.v_sim_setting['link_attrs_setting']: l_attr['high'] = config.v_sim_setting_high_request
        for n_attr in config.v_sim_setting['node_attrs_setting']: n_attr['low'] = config.v_sim_setting_low_request
        for l_attr in config.v_sim_setting['link_attrs_setting']: l_attr['low'] = config.v_sim_setting_low_request

    for key, value in adjust_v_sim_setting.items():
        if isinstance(key, dict):
            config.v_sim_setting[key].update(value)
        else:
            config.v_sim_setting[key] = value

    # get dataset dir
    config.num_p_net_node_attrs = len(config.p_net_setting['node_attrs_setting'])
    config.num_p_net_link_attrs = len(config.p_net_setting['link_attrs_setting'])
    config.num_v_net_node_attrs = len(config.v_sim_setting['node_attrs_setting'])
    config.num_v_net_link_attrs = len(config.v_sim_setting['link_attrs_setting'])

    # host and time 
    config.run_time = time.strftime('%Y%m%dT%H%M%S')
    config.host_name = socket.gethostname()
    config.run_id = f'{config.host_name}-{config.run_time}'

    if config.target_steps == -1:
        config.target_steps = config.batch_size

    check_config(config)
    return config

def check_config(config):
    # check config
    assert config.reusable == False, 'Unsupported currently!'
    if config.target_steps != -1: assert config.target_steps % config.batch_size == 0, 'A should be greater than b!'

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
        config = get_config()
        print(f'Load default config')
    return config

def delete_empty_dir(config):
    for dir in [config.record_dir, config.log_dir, config.save_dir]:
        if os.path.exists(dir) and not os.listdir(dir):
            os.rmdir(dir)


if __name__ == "__main__":
    get_config()