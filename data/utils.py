import os
import json
import yaml
import numpy as np
import networkx as nx


def path_to_links(path):
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

def get_distribution_average(self, distribution, dtype, **kwargs):
    pass

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