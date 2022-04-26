import os
import json
import yaml
import numpy as np
import networkx as nx


def read_setting(fpath, mode='r'):
    with open(fpath, mode) as f:
        if fpath[-4:] == 'json':
            setting_dict = json.load(f)
        elif fpath[-4:] == 'yaml':
            setting_dict = yaml.load(f, Loader=yaml.Loader)
        else:
            return ValueError('Only supports settings files in yaml and json format!')
    return setting_dict

def write_setting(setting_dict, fpath, mode='w+'):
    with open(fpath, mode) as f:
        if fpath[-4:] == 'json':
            json.dump(setting_dict, f)
        elif fpath[-4:] == 'yaml':
            yaml.dump(setting_dict, f)
        else:
            return ValueError('Only supports settings files in yaml and json format!')
    return setting_dict

def path_to_edges(path):
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def generate_data_with_distribution(size, distribution, dtype, **kwargs):
    assert distribution in ['uniform', 'normal', 'exponential', 'possion']
    assert dtype in ['int', 'float', 'bool']
    if distribution == 'normal':
        loc, scale = kwargs.get('loc'), kwargs.get('scale')
        data = np.random.normal(loc, scale, size)
    elif distribution == 'uniform':
        low, high = kwargs.get('low'), kwargs.get('high')
        if dtype == 'int':
            data = np.random.randint(low, high + 1, size)
        elif dtype == 'float':
            data = np.random.uniform(low, high, size)
    elif distribution == 'exponential':
        scale = kwargs.get('scale')
        data = np.random.exponential(scale, size)
    elif distribution == 'possion':
        lam = kwargs.get('lam')
        if kwargs.get('reciprocal', False):
            lam = 1 / lam
        data = np.random.poisson(lam, size)
    else:
        raise NotImplementedError(f'Generating {dtype} data following the {distribution} distribution is unsupporrted!')
    return data.astype(dtype).tolist()

def draw_graph(G, width=0.05, show=True, save_path=None):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8), dpi=200)
    size = 10
    edge_colors = None
    options = {
        'pos': pos, 
        "node_color": 'red',
        "node_size": size,
        # "line_color": "grey",
        "linewidths": 0,
        "width": width,
        # 'with_label': True, 
        "cmap": plt.cm.brg,
        'edge_color': edge_colors,
        'edge_cmap': plt.cm.Blues, 
        'alpha': 0.5, 
    }
    nx.draw(G, **options)
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
