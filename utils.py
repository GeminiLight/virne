import os
import json
import yaml
import shutil
from functools import wraps


def read_setting(fpath, mode='r'):
    with open(fpath, mode, encoding='utf-8') as f:
        if fpath[-4:] == 'json':
            setting_dict = json.load(f)
        elif fpath[-4:] == 'yaml':
            setting_dict = yaml.load(f, Loader=yaml.Loader)
        else:
            return ValueError('Only supports settings files in yaml and json format!')
    return setting_dict

def write_setting(setting_dict, fpath, mode='w'):
    with open(fpath, mode, encoding='utf-8') as f:
        if fpath[-4:] == 'json':
            json.dump(setting_dict, f)
        elif fpath[-4:] == 'yaml':
            yaml.dump(setting_dict, f)
        else:
            return ValueError('Only supports settings files in yaml and json format!')
    return setting_dict

def conver_format(fpath_1, fpath_2):
    setting_dict = read_setting(fpath_1)
    write_setting(setting_dict, fpath_2)

def generate_file_name(config, epoch_id=0, extra_items=[], **kwargs):
    if not isinstance(config, dict): config = vars(config)
    items = extra_items + ['p_net_num_nodes', 'reusable']

    file_name_1 = f"{config['solver_name']}-records-{epoch_id}-"
    # file_name_2 = '-'.join([f'{k}={config[k]}' for k in items])
    file_name_3 = '-'.join([f'{k}={v}' for k, v in kwargs.items()])
    file_name = file_name_1 + file_name_3 + '.csv'
    return file_name

def get_p_net_dataset_dir_from_setting(p_net_setting):
    p_net_dataset_dir = p_net_setting.get('save_dir')
    n_attrs = [n_attr['name'] for n_attr in p_net_setting['node_attrs_setting']]
    e_attrs = [l_attr['name'] for l_attr in p_net_setting['link_attrs_setting']]

    if 'file_path' in p_net_setting['topology'] and os.path.exists(p_net_setting['topology']['file_path']):
        p_net_name = f"{os.path.basename(p_net_setting['topology']['file_path']).split('.')[0]}"
    else:
        p_net_name = f"{p_net_setting['num_nodes']}-{p_net_setting['topology']['type']}_[{p_net_setting['topology']['wm_alpha']}-{p_net_setting['topology']['wm_beta']}]"
    node_attrs_str = '-'.join([f'{n_attr_setting["name"]}_{get_parameters_string(get_distribution_parameters(n_attr_setting))}' for n_attr_setting in p_net_setting['node_attrs_setting']])
    link_attrs_str = '-'.join([f'{e_attr_setting["name"]}_{get_parameters_string(get_distribution_parameters(e_attr_setting))}' for e_attr_setting in p_net_setting['link_attrs_setting']])
    
    p_net_dataset_middir = p_net_name + '-' + node_attrs_str + '-' + link_attrs_str
                        # f"{n_attrs}-[{p_net_setting['node_attrs_setting'][0]['low']}-{p_net_setting['node_attrs_setting'][0]['high']}]-" + \
                        # f"{e_attrs}-[{p_net_setting['link_attrs_setting'][0]['low']}-{p_net_setting['link_attrs_setting'][0]['high']}]"        
    p_net_dataset_dir = os.path.join(p_net_dataset_dir, p_net_dataset_middir)
    return p_net_dataset_dir

def get_v_nets_dataset_dir_from_setting(v_sim_setting):
    v_nets_dataset_dir = v_sim_setting.get('save_dir')
    # n_attrs = [n_attr['name'] for n_attr in v_sim_setting['node_attrs_setting']]
    # e_attrs = [l_attr['name'] for l_attr in v_sim_setting['link_attrs_setting']]
    node_attrs_str = '-'.join([f'{n_attr_setting["name"]}_{get_parameters_string(get_distribution_parameters(n_attr_setting))}' for n_attr_setting in v_sim_setting['node_attrs_setting']])
    link_attrs_str = '-'.join([f'{e_attr_setting["name"]}_{get_parameters_string(get_distribution_parameters(e_attr_setting))}' for e_attr_setting in v_sim_setting['link_attrs_setting']])
    
    v_nets_dataset_middir = f"{v_sim_setting['num_v_nets']}-[{v_sim_setting['v_net_size']['low']}-{v_sim_setting['v_net_size']['high']}]-" + \
                        f"{v_sim_setting['topology']['type']}-{get_parameters_string(get_distribution_parameters(v_sim_setting['lifetime']))}-{v_sim_setting['arrival_rate']['lam']}-" + \
                        node_attrs_str + '-' + link_attrs_str
                        # f"{n_attrs}-[{v_sim_setting['node_attrs_setting'][0]['low']}-{v_sim_setting['node_attrs_setting'][0]['high']}]-" + \
                        # f"{e_attrs}-[{v_sim_setting['link_attrs_setting'][0]['low']}-{v_sim_setting['link_attrs_setting'][0]['high']}]"
    v_net_dataset_dir = os.path.join(v_nets_dataset_dir, v_nets_dataset_middir)
    return v_net_dataset_dir

def get_distribution_parameters(distribution_dict):
    distribution = distribution_dict.get('distribution', None)
    if distribution is None:
        return []
    if distribution == 'exponential':
        parameters = [distribution_dict['scale']]
    elif distribution == 'possion':
        parameters = [distribution_dict['lam']]
    elif distribution == 'uniform':
        parameters = [distribution_dict['low'], distribution_dict['high']]
    elif distribution == 'customized':
        parameters = [distribution_dict['min'], distribution_dict['max']]
    return parameters

def get_parameters_string(parameters):
    if len(parameters) == 0:
        return 'None'
    elif len(parameters) == 1:
        return str(parameters[0])
    else:
        str_parameters = [str(p) for p in parameters]
        return f'[{"-".join(str_parameters)}]'

def delete_temp_files(file_path):
    del_list = os.listdir(file_path)
    for f in del_list:
        file_path = os.path.join(del_list, f)
        if os.path.isfile(file_path) and 'temp' in file_path:
            os.remove(file_path)

def clean_save_dir(dir):
    sub_dirs = ['model', 'records', 'log']
    algo_dir_list = [os.path.join(dir, algo_name) for algo_name in os.listdir(dir) if os.path.isdir(os.path.join(dir, algo_name))]
    for algo_dir in algo_dir_list:
        for run_id in os.listdir(algo_dir):
            run_id_dir = os.path.join(algo_dir, run_id)
            record_dir = os.path.join(run_id_dir, 'records')
            if not os.path.exists(record_dir) or not os.listdir(record_dir):
                shutil.rmtree(run_id_dir)
                print(f'Delate {run_id_dir}')


def test_running_time(func):
    import time
    @wraps(func)
    def test(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print(f'Running time of {func.__name__}: {t2-t1:2.4f}s')
        return res
    return test