try:
    import torch_sparse, torch_scatter, torch_cluster
except ImportError:
    print('PyTorch Geometric is not installed completely. Installing now...')
    import os
    import torch
    cuda_version = torch.version.cuda
    # if cuda_version is None:
    #     cuda_suffix = 'cpu'
    if cuda_version in [11.7, 11.8, 12.1]:
        cuda_suffix = 'cu' + cuda_version.replace('.', '')
    else:
        cuda_suffix = 'cpu'
    os.system(f'pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+{cuda_suffix}.html')
    # os.system('cls' if os.name == 'nt' else 'clear')


from .hopfield_network import HopfieldNetworkSolver
from .gae_clustering import GaeClusteringSolver
from .mcts import MctsSolver
from .pg_mlp import PgMlpSolver
from .pg_cnn import PgCnnSolver
from .pg_cnn2 import PgCnn2Solver
from .ddpg_attention import DdpgAttentionSolver
from .pg_seq2seq import PgSeq2SeqSolver
from .a3c_gcn_seq2seq import A3CGcnSeq2SeqSolver


__all__ = [
    # Unsupervised learning solvers
    'HopfieldNetworkSolver',
    'GaeClusteringSolver',
    # Reinforcement learning solvers
    'MctsSolver',
    'PgMlpSolver',
    'PgCnnSolver',
    'PgCnn2Solver',
    'PgSeq2SeqSolver',
    'A3CGcnSeq2SeqSolver',
    'DdpgAttentionSolver'
]