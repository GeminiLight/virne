import os
import json
import time
import pprint
import socket
import argparse

from utils import read_setting, get_pn_dataset_dir_from_setting, get_vns_dataset_dir_from_setting, write_setting


parser = argparse.ArgumentParser(description='configuration file')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

### Dataset ###
data_arg = parser.add_argument_group('data')
data_arg.add_argument('--pn_setting_path', type=str, default='settings/pn_setting.yaml', help='File path of physical network settings')
data_arg.add_argument('--vns_setting_path', type=str, default='settings/vns_setting.yaml', help='File path of virtual network settings')

data_arg.add_argument('--if_adjust_vns_setting', type=str2bool, default=False, help='Whether to deirectly adjust VN settings in parser')
data_arg.add_argument('--vns_setting_max_length', type=int, default=10, help='')
data_arg.add_argument('--vns_setting_aver_arrival_rate', type=float, default=0.10, help='')
data_arg.add_argument('--vns_setting_aver_lifetime', type=int, default=1000, help='')
data_arg.add_argument('--vns_setting_high_request', type=int, default=30, help='')
data_arg.add_argument('--vns_setting_low_request', type=int, default=0, help='')
data_arg.add_argument('--vns_setting_num_vns', type=int, default=1000, help='')

### Environment ###
env_arg = parser.add_argument_group('env')
env_arg.add_argument('--summary_dir', type=str, default='records/', help='File Directory to save summary and records')
env_arg.add_argument('--summary_file_name', type=str, default='global_summary.csv', help='Summary file name')
env_arg.add_argument('--if_save_records', type=str2bool, default=True, help='Whether to save records')
env_arg.add_argument('--if_temp_save_records', type=str2bool, default=True, help='Whether to temporarily save records')

### solver  ###
solver_arg = parser.add_argument_group('solver')
solver_arg.add_argument('--verbose', type=str2bool, default=1, help='Level of showing information')
solver_arg.add_argument('--solver_name', type=str, default='ns_gnn', help='Name of the solver selected to run')
solver_arg.add_argument('--reusable', type=str2bool, default=False, help='Whether or not to allow to deploy several VN nodes on the same VNF')

### Neural Network ###
net_arg = parser.add_argument_group('net')
# device
net_arg.add_argument('--use_cuda', type=str2bool, default=True, help='Whether to sse GPU to accelerate the training process')
net_arg.add_argument('--cuda_id', type=int, default=0, help='CUDA device id')
net_arg.add_argument('--allow_parallel', type=str2bool, default=False, help='Whether to use mutiple GPUs')
# nn
net_arg.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
net_arg.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
net_arg.add_argument('--num_layers', type=int, default=1, help='Number of GRU stacks\' layers')
net_arg.add_argument('--num_gnn_layers', type=int, default=5, help='Number of GNN layers')
net_arg.add_argument('--dropout_prob', type=float, default=0.0, help='Droput rate')
net_arg.add_argument('--batch_norm', type=float, default=False, help='Droput rate')
net_arg.add_argument('--l2reg_rate', type=float, default=2.5e-4, help='L2 regularization rate')

### Reinforcement Learning ###
rl_arg = parser.add_argument_group('net')
# rl
rl_arg.add_argument('--rl_gamma', type=float, default=0.99, help='Cumulative reward discount rate')
rl_arg.add_argument('--explore_rate', type=float, default=0.9, help='Epsilon-greedy explore rate')
rl_arg.add_argument('--lr', type=float, default=1e-3, help='General Learning rate')
rl_arg.add_argument('--lr_actor', type=float, default=1e-3, help='Actor learning rate')
rl_arg.add_argument('--lr_critic', type=float, default=1e-3, help='Critic learning rate')
rl_arg.add_argument('--weight_decay', type=float, default=5e-4, help='')
rl_arg.add_argument('--coef_value_loss', type=float, default=0.5, help='')
rl_arg.add_argument('--coef_entropy_loss', type=float, default=0.01, help='')
rl_arg.add_argument('--coef_mask_loss', type=float, default=0.000, help='')
# tricks
rl_arg.add_argument('--max_grad_norm', type=float, default=0.5, help='')
rl_arg.add_argument('--maskable_policy', type=str2bool, default=True, help='')
rl_arg.add_argument('--norm_advantage', type=str2bool, default=True, help='')
rl_arg.add_argument('--clip_grad', type=str2bool, default=True, help='')
rl_arg.add_argument('--norm_value_loss', type=str2bool, default=True, help='')

### Trainning ###
train_arg = parser.add_argument_group('train')
train_arg.add_argument('--num_train_epochs', type=int, default=100, help='Number of epochs')
train_arg.add_argument('--num_workers', type=int, default=10, help='Number of sub workers who collect experience asynchronously')
train_arg.add_argument('--batch_size', type=int, default=256, help='Batch size of training')
train_arg.add_argument('--target_steps', type=int, default=-1, help='Target steps for collecting experience')
train_arg.add_argument('--repeat_times', type=int, default=10, help='')
train_arg.add_argument('--gae_lambda', type=float, default=0.98, help='')
train_arg.add_argument('--eps_clip', type=float, default=0.2, help='')
train_arg.add_argument('--use_negative_sample', type=str2bool, default=False, help='Whether to allow use failed sample to train')

### Run ###
run_arg = parser.add_argument_group('run')
run_arg.add_argument('--if_save_config', type=str2bool, default=True, help='Whether to save config')
run_arg.add_argument('--only_test', type=str2bool, default=False, help='Only test without training')
run_arg.add_argument('--renew_vn_simulator', type=str2bool, default=False, help='')
run_arg.add_argument('--start_epoch', type=int, default=0, help='Start from epochi')
run_arg.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
run_arg.add_argument('--random_seed', type=int, default=1111, help='Random seed')
run_arg.add_argument('--save_dir', type=str, default='save', help='Save directory for models and trainning logs')
run_arg.add_argument('--pretrained_model_path', type=str, default='', help='')
run_arg.add_argument('--open_tb', type=str2bool, default=True, help='')
run_arg.add_argument('--log_interval', type=int, default=1, help='')
run_arg.add_argument('--save_interval', type=int, default=10, help='')
run_arg.add_argument('--eval_interval', type=int, default=10, help='')
run_arg.add_argument('--total_steps', type=int, default=5000000, help='')
run_arg.add_argument('--sub_solver', type=str, default='nrm_rank', help='')
run_arg.add_argument('--reward_weight', type=float, default=0.1, help='')

# run_arg.add_argument('--save_model', type=str2bool, default=False, help='Save model')
# run_arg.add_argument('--load_model', type=str2bool, default=False, help='Load model')
#run_arg.add_argument('--lr_decay_step', type=int, default=5000, help='Lr1 decay step')
#run_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='Lr1 decay rate')

### Misc ###
misc_arg = parser.add_argument_group('misc')


def get_config(args=None, adjust_pn_setting={}, adjust_vns_setting={}):
    config = parser.parse_args(args)

    if config.target_steps == -1:
        config.target_steps = config.batch_size

    if config.reusable:
        print('*' * 40)
        print(' ' * 5 + '!!! Physical Node Hosts is Reusable !!!')
        print('*' * 40)

    # make dir
    for dir in [config.save_dir, config.summary_dir, 'dataset', 'dataset/pn', 'dataset/vns']:
        if not os.path.exists(dir):
            os.makedirs(dir)
            
    # read pn and vns setting
    config.pn_setting = read_setting(config.pn_setting_path)
    config.vns_setting = read_setting(config.vns_setting_path)

    config.pn_setting.update(adjust_pn_setting)
    config.vns_setting.update(adjust_vns_setting)

    if config.if_adjust_vns_setting:
        config.vns_setting['max_length'] = config.vns_setting_max_length
        config.vns_setting['aver_arrival_rate'] = config.vns_setting_aver_arrival_rate
        config.vns_setting['aver_lifetime'] = config.vns_setting_aver_lifetime
        for n_attr in config.vns_setting['node_attrs_setting']: n_attr['high'] = config.vns_setting_high_request
        for e_attr in config.vns_setting['edge_attrs_setting']: e_attr['high'] = config.vns_setting_high_request
        for n_attr in config.vns_setting['node_attrs_setting']: n_attr['low'] = config.vns_setting_low_request
        for e_attr in config.vns_setting['edge_attrs_setting']: e_attr['low'] = config.vns_setting_low_request


    # if adjust_pn_setting != {}:
    #     assert NotImplementedError
    for key, value in adjust_vns_setting.items():
        if isinstance(key, dict):
            config.vns_setting[key].update(value)
        else:
            config.vns_setting[key] = value

        # if item == 'max_length':
        #     config.vns_setting['vn_size']['high'] = value
        # elif item == 'min_length':
        #     config.vns_setting['vn_size']['low'] = value
        # elif item == 'high_request':
        #     for n_attr in config.vns_setting['node_attrs_setting']: n_attr['high'] = value
        #     for e_attr in config.vns_setting['edge_attrs_setting']: e_attr['high'] = value
        # elif item == 'low_request':
        #     for n_attr in config.vns_setting['node_attrs_setting']: n_attr['low'] = value
        #     for e_attr in config.vns_setting['edge_attrs_setting']: e_attr['low'] = value
        # else:
        #     config.vns_setting[item] = value

    # get dataset dir
    config.pn_dataset_dir = get_pn_dataset_dir_from_setting(config.pn_setting)
    config.vns_dataset_dir = get_vns_dataset_dir_from_setting(config.vns_setting)
    config.num_pn_node_attrs = len(config.pn_setting['node_attrs_setting'])
    config.num_pn_edge_attrs = len(config.pn_setting['edge_attrs_setting'])
    config.num_vn_node_attrs = len(config.vns_setting['node_attrs_setting'])
    config.num_vn_edge_attrs = len(config.vns_setting['edge_attrs_setting'])

    # host and time 
    config.run_time = time.strftime('%Y%m%dT%H%M%S')
    config.host_name = socket.gethostname()
    config.run_id = f'{config.host_name}-{config.run_time}'

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
    pass