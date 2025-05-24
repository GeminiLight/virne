try:
    import torch_sparse, torch_scatter, torch_cluster
except ImportError:
    # raise ImportError('Please install PyTorch Geometric first.')
    print('PyTorch Geometric is not installed completely. Installing now...')
    import os
    import torch
    cuda_version = torch.version.cuda
    if cuda_version is None:
        cuda_suffix = 'cpu'
    elif cuda_version in [12.4]:
        cuda_suffix = 'cu' + cuda_version.replace('.', '')
    
    os.system(f'pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+{cuda_suffix}.html')
    os.system('cls' if os.name == 'nt' else 'clear')


from .reinforcement_learning import *
from .unsupervised_learning import *



__all__ = [
]