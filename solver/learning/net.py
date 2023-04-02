from itertools import chain
import math
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, PNAConv, NNConv, SAGEConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from .graph_conv import EdgeFusionGATConv


class MLPNet(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers=2, embedding_dims=None, batch_norm=False, dropout_prob=1.0):
        super(MLPNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        if embedding_dims is not None: 
            assert len(embedding_dims) == num_layers - 1, 'len(embedding_dims) should equal to num_layers-1'
        else:
            embedding_dims = [self.input_dim * 2] * (num_layers-1)
        sequential = []
        for layer_id in range(self.num_layers):
            if layer_id == 0:
                lin = nn.Linear(self.input_dim, embedding_dims[layer_id])
                norm = nn.BatchNorm1d(embedding_dims[layer_id]) if batch_norm else nn.Identity()
            elif layer_id == self.num_layers - 1:
                lin = nn.Linear(embedding_dims[-1], output_dim)
                norm = nn.Identity()
            else:
                lin = nn.Linear(embedding_dims[layer_id-1], embedding_dims[layer_id])
                norm = nn.BatchNorm1d(embedding_dims[layer_id]) if batch_norm else nn.Identity()
            sequential += [lin, norm]
            if layer_id != self.num_layers - 1:
                sequential += [nn.LeakyReLU()]
        self.lins = nn.Sequential(*sequential)

        self._init_parameters()

    def _init_parameters(self):
        for mod in self.lins:
            if isinstance(mod, nn.Linear):
                nn.init.orthogonal_(mod.weight)

    def forward(self, input):
        return self.lins(input)


class PositionalEncoder(nn.Module):

    def __init__(self, embedding_dim, dropout_prob=1.0, max_len=50, method='add'):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob < 1. else nn.Identity()
        # Compute the positional encodingss
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.method = method
        
    def forward(self, x):
        pe_embeddings = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        pe_embeddings = pe_embeddings.repeat(x.shape[0], 1, 1)
        # if len(x.shape) == 3:
            # pe_embeddings = pe_embeddings.unsqueeze(0).repeat(x.shape[0], 1, 1)
        if self.method == 'add':
            x = x + pe_embeddings
        elif self.method == 'concat':
            x = torch.concat([x, pe_embeddings], dim=-1)
        return self.dropout(x)


class ResNetBlock(nn.Module):
    
    def __init__(self, n_input_channels):
        super(ResNetBlock, self).__init__()
        self.conv_1 = nn.Conv2d(n_input_channels, n_input_channels, kernel_size=[1, 1], stride=[1, 1])
        self.conv_2 = nn.Conv2d(n_input_channels, n_input_channels, kernel_size=[1, 1], stride=[1, 1])

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.conv_1(x))
        out = F.leaky_relu(self.conv_2(out))
        out = out + identity
        return out


class ResLinearNet(nn.Module):

    def __init__(self, num_in_feats, num_out_feats, num_layers=3, batch_norm=True):
        super(ResNetBlock, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_layers = num_layers

        self.first_linear = None
        self.last_linear = None
        self.sequential = []
        self.output_seq = []

        for l in range(self.num_layers):
            if l == 0:
                self.first_linear = nn.Linear(self.num_in_feats, self.num_out_feats)
                if batch_norm: self.sequential.append(nn.BatchNorm1d(self.num_out_feats))
                self.sequential.append(nn.LeakyReLU())
            elif l == self.num_layers - 1:
                self.last_linear = nn.Linear(self.num_out_feats, self.num_out_feats)
                if batch_norm: self.output_seq.append(nn.BatchNorm1d(self.num_out_feats))
            else:
                self.sequential.append(nn.Linear(self.num_out_feats, self.num_out_feats))
                if batch_norm: self.sequential.append(nn.BatchNorm1d(self.num_out_feats))
                self.sequential.append(nn.LeakyReLU())

        self.sequential = nn.Sequential(*self.sequential)
        self.output_seq = nn.Sequential(*self.output_seq)

        self.init_parameters()

    def init_parameters(self):
        for mod in chain(self.sequential, self.output_seq):
            if isinstance(mod, nn.Linear):
                nn.init.orthogonal_(mod.weight)

    def forward(self, inp):
        x1 = self.first_linear(inp)
        x2 = self.sequential(x1) + x1
        return self.output_seq(x2)


class GraphConvNet(nn.Module):
    """Graph Convolutional Network to extract the feature of physical network."""
    def __init__(self, input_dim, output_dim, num_layers=3, embedding_dim=128, num_heads=1, edge_dim=None, batch_norm=False, dropout_prob=1.0, return_batch=False, pooling=None):
        super(GraphConvNet, self).__init__()
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.return_batch = return_batch
        self.pooling = pooling
        if self.pooling is not None:
            self.graph_pooling = GraphPooling(aggr=pooling, output_dim=output_dim)


        for layer_id in range(self.num_layers):
            if self.num_layers == 1:
                conv = self.get_conv(input_dim, output_dim, heads=num_heads, edge_dim=edge_dim, aggr='add', bias=True)
            elif self.num_layers == 1:
                conv = self.get_conv(input_dim, output_dim, heads=num_heads, edge_dim=edge_dim, aggr='add', bias=True)
            elif layer_id == 0:
                conv = self.get_conv(input_dim, embedding_dim, heads=num_heads, edge_dim=edge_dim, aggr='add', bias=True)
            elif layer_id == num_layers - 1:
                conv = self.get_conv(embedding_dim, output_dim, heads=num_heads, edge_dim=edge_dim, aggr='add', bias=True)
            else:
                conv = self.get_conv(embedding_dim, embedding_dim, heads=num_heads, edge_dim=edge_dim, aggr='add', bias=True)
            
            norm_dim = output_dim if layer_id == num_layers - 1 else embedding_dim
            norm = nn.BatchNorm1d(norm_dim) if batch_norm else nn.Identity()
            dout = nn.Dropout(dropout_prob) if dropout_prob < 1. else nn.Identity()

            self.add_module('conv_{}'.format(layer_id), conv)
            self.add_module('norm_{}'.format(layer_id), norm)
            self.add_module('dout_{}'.format(layer_id), dout)

        self._init_parameters()

    def get_conv(self, input_dim, output_dim, edge_dim=None, aggr='add', bias=True, **kwargs):
        return NotImplementedError

    def _init_parameters(self):
        for layer_id in range(self.num_layers):
            nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_src.weight)
            nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_dst.weight)
            if self.edge_dim is not None:
                nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_edge.weight)

    def forward(self, input):
        x, edge_index, edge_attr = input['x'], input['edge_index'], input.get('edge_attr', None)
        for layer_id in range(self.num_layers):
            conv = getattr(self, 'conv_{}'.format(layer_id))
            norm = getattr(self, 'norm_{}'.format(layer_id))
            dout = getattr(self, 'dout_{}'.format(layer_id))
            x = conv(x, edge_index, edge_attr)
            if layer_id == self.num_layers - 1:
                x = dout(norm(x))
            else:
                x = F.leaky_relu(dout(norm(x)))
        if self.return_batch:
            x, mask = to_dense_batch(x, input.batch)
        else:
            if self.pooling is not None:
                x = self.graph_pooling(x, input.batch)
        return x


class GATConvNet(GraphConvNet):
    
    def __init__(self, input_dim, output_dim, num_layers=3, embedding_dim=128, num_heads=1, edge_dim=None, batch_norm=False, dropout_prob=1.0, return_batch=False, pooling=None):
        super(GATConvNet, self).__init__(input_dim, output_dim, num_layers, embedding_dim, num_heads, edge_dim, batch_norm, dropout_prob, return_batch, pooling)

    def get_conv(self, input_dim, output_dim, edge_dim=None, aggr='add', bias=True, **kwargs):
        num_heads = kwargs.get('num_heads', 1)
        fill_value = kwargs.get('fill_value', 'max')
        return GATConv(input_dim, output_dim, heads=num_heads, edge_dim=edge_dim, aggr=aggr, bias=bias, fill_value=fill_value)

class EdgeFusionGATConvNet(GraphConvNet):
    
    def __init__(self, input_dim, output_dim, num_layers=3, embedding_dim=128, num_heads=1, edge_dim=None, batch_norm=False, dropout_prob=1.0, return_batch=False, pooling=None):
        super(EdgeFusionGATConvNet, self).__init__(input_dim, output_dim, num_layers, embedding_dim, num_heads, edge_dim, batch_norm, dropout_prob, return_batch, pooling)

    def get_conv(self, input_dim, output_dim, edge_dim=None, aggr='add', bias=True, **kwargs):
        num_heads = kwargs.get('num_heads', 1)
        fill_value = kwargs.get('fill_value', 'max')
        return EdgeFusionGATConv(input_dim, output_dim, heads=num_heads, edge_dim=edge_dim, aggr=aggr, bias=bias, fill_value=fill_value)


class NNConvNet(GraphConvNet):
    
    def __init__(self, input_dim, output_dim, num_layers=3, embedding_dim=128, num_heads=1, edge_dim=None, batch_norm=False, dropout_prob=1.0, return_batch=False, pooling=None):
        super(NNConvNet, self).__init__(input_dim, output_dim, num_layers, embedding_dim, num_heads, edge_dim, batch_norm, dropout_prob, return_batch, pooling)

    def get_conv(self, input_dim, output_dim, edge_dim=None, aggr='add', bias=True, **kwargs):
        return NNConv(input_dim, output_dim, aggr=aggr, bias=bias)

class PNAConvNet(GraphConvNet):
    
    def __init__(self, input_dim, output_dim, num_layers=3, embedding_dim=128, num_heads=1, edge_dim=None, batch_norm=False, dropout_prob=1.0, return_batch=False, pooling=None):
        super(PNAConvNet, self).__init__(PNAConv, input_dim, output_dim, num_layers, embedding_dim, num_heads, edge_dim, batch_norm, dropout_prob, return_batch, pooling)

    def get_conv(self, input_dim, output_dim, edge_dim=None, aggr='add', bias=True, **kwargs):
        return PNAConv(input_dim, output_dim, aggregators=['sum', 'max', 'min'], scalers='identity', edge_dim=edge_dim, bias=bias)


class GCNConvNet(nn.Module):
    """Graph Convolutional Network to extract the feature of physical network."""
    def __init__(self, input_dim, output_dim, embedding_dim=128, num_layers=3, batch_norm=True, dropout_prob=1.0, return_batch=False, pooling=None, **kwargs):
        super(GCNConvNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.return_batch = return_batch
        self.pooling = pooling
        if self.pooling is not None:
            self.graph_pooling = GraphPooling(aggr=pooling, output_dim=output_dim)


        for layer_id in range(self.num_layers):
            if self.num_layers == 1:
                conv = GCNConv(input_dim, output_dim)
            elif layer_id == 0:
                conv = GCNConv(input_dim, embedding_dim)
            elif layer_id == num_layers - 1:
                conv = GCNConv(embedding_dim, output_dim)
            else:
                conv = GCNConv(embedding_dim, embedding_dim)
                
            norm_dim = output_dim if layer_id == num_layers - 1 else embedding_dim
            norm = nn.BatchNorm1d(norm_dim) if batch_norm else nn.Identity()
            dout = nn.Dropout(dropout_prob) if dropout_prob < 1. else nn.Identity()

            self.add_module('conv_{}'.format(layer_id), conv)
            self.add_module('norm_{}'.format(layer_id), norm)
            self.add_module('dout_{}'.format(layer_id), dout)

        self._init_parameters()

    def _init_parameters(self):
        for layer_id in range(self.num_layers):
            nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin.weight)

    def forward(self, input):
        x, edge_index = input['x'], input['edge_index']
        for layer_id in range(self.num_layers):
            conv = getattr(self, 'conv_{}'.format(layer_id))
            norm = getattr(self, 'norm_{}'.format(layer_id))
            dout = getattr(self, 'dout_{}'.format(layer_id))
            x = conv(x, edge_index)
            if layer_id == self.num_layers - 1:
                x = dout(norm(x))
            else:
                x = F.leaky_relu(dout(norm(x)))
        if self.return_batch:
            x, mask = to_dense_batch(x, input.batch)
        if self.pooling is not None:
            x = self.graph_pooling(x, input.batch)
        return x


class DeepEdgeFeatureGAT(nn.Module):
    """five layers"""
    def __init__(self, input_dim, output_dim, edge_dim, num_layers=5, alpha=0.2, theta=0.2, embedding_dim=128, num_heads=1, batch_norm=False, dropout_prob=1.0, return_batch=False, pooling=None):
        super(DeepEdgeFeatureGAT, self).__init__()
        assert num_layers >= 2
        self.alpha = alpha
        self.theta = theta
        self.edge_dim = edge_dim
        self.num_mid_layers = num_layers - 2
        self.return_batch = return_batch
        self.pooling = pooling
        if self.pooling is not None:
            self.graph_pooling = GraphPooling(aggr=pooling, output_dim=output_dim)

        self.conv_s = GATConv(input_dim, embedding_dim, heads=num_heads, edge_dim=edge_dim)
        for layer_id in range(self.num_mid_layers):
            conv = GATConv(embedding_dim, embedding_dim, heads=num_heads, edge_dim=edge_dim)
            norm = nn.BatchNorm1d(embedding_dim) if batch_norm else nn.Identity()
            dout = nn.Dropout(dropout_prob) if dropout_prob < 1. else nn.Identity()
            weight = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
            self.add_module('conv_{}'.format(layer_id), conv)
            self.add_module('norm_{}'.format(layer_id), norm)
            self.add_module('dout_{}'.format(layer_id), dout)
            self.register_parameter(f'weight_{layer_id}', weight)
        self.conv_e = GATConv(embedding_dim, output_dim, heads=num_heads, edge_dim=edge_dim)
        self._init_parameters()

    def _init_parameters(self):
        for layer_id in list(range(self.num_mid_layers)) + ['s', 'e']:
            nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_src.weight)
            nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_dst.weight)
            if self.edge_dim is not None:
                nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_edge.weight)
            if layer_id not in ['s', 'e']:
                nn.init.orthogonal_(getattr(self, f'weight_{layer_id}'))

    def forward(self, input):
        x, edge_index, edge_attr = input['x'], input['edge_index'], input.get('edge_attr', None)
        x_0 = self.conv_s(x, edge_index, edge_attr)
        x = x_0
        for layer_id in range(self.num_mid_layers):
            conv = getattr(self, 'conv_{}'.format(layer_id))
            norm = getattr(self, 'norm_{}'.format(layer_id))
            dout = getattr(self, 'dout_{}'.format(layer_id))
            weight = getattr(self, f'weight_{layer_id}')
            conv_x = conv(x, edge_index, edge_attr)
            # x | initial residual | identity mapping
            beta = math.log(self.theta / (layer_id + 1) + 1)
            conv_x.mul_(1 - self.alpha)
            res_x = self.alpha * x_0
            x = conv_x.add_(res_x)
            x = torch.addmm(x, x, weight, beta=1. - beta, alpha=beta)
            x = F.leaky_relu(dout(norm(x)))
        x = self.conv_e(x, edge_index, edge_attr)
        if self.return_batch:
            x, mask = to_dense_batch(x, input.batch)
        if self.pooling is not None:
            x = self.graph_pooling(x, input.batch)
        return x


class GraphPooling(nn.Module):

    def __init__(self, aggr='sum', **kwargs):
        super(GraphPooling, self).__init__()
        if aggr in ['att', 'attention']:
            output_dim = kwargs.get('output_dim')
            self.pooling = GraphAttentionPooling(output_dim)
        elif aggr in ['add', 'sum']:
            self.pooling = global_add_pool
        elif aggr == 'max':
            self.pooling = global_max_pool
        elif aggr == 'mean':
            self.pooling = global_mean_pool
        else:
            return NotImplementedError

    def forward(self, x, batch):
        return self.pooling(x, batch)


class GraphAttentionPooling(nn.Module):
    """Attention module to extract global feature of a graph."""
    def __init__(self, input_dim):
        super(GraphAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.input_dim))
        self._init_parameters()

    def _init_parameters(self):
        """Initializing weights."""
        nn.init.orthogonal_(self.weight)

    def forward(self, x, batch, size=None):
        """
        Making a forward propagation pass to create a graph level representation.

        Args:
            x (torch.Tensor): Result of the GNN.
            batch (torch.Tensor): Batch vector, which assigns each node to a specific example
            size (int, optional): Number of nodes in the graph. Defaults to None.

        Returns:
            representation: A graph level representation matrix.
        """
        size = batch[-1].item() + 1 if size is None else size
        mean = scatter(x, batch, dim=0, dim_size=size, reduce='mean')
        transformed_global = torch.tanh(torch.mm(mean, self.weight))

        coefs = torch.sigmoid((x * transformed_global[batch] * 10).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return scatter(weighted, batch, dim=0, dim_size=size, reduce='add')

    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight))

        return torch.sigmoid(torch.matmul(x, transformed_global))


class MultiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            num_heads=1,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.n_heads = num_heads

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // self.n_heads
        if key_dim is None:
            key_dim = val_dim
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(self.n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(self.n_heads, input_dim, key_dim))
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None):
        """

        :param q: queries (batch_size, num_query, input_dim)
        :param exchange: (batch_size, 2)
        :return:
        """
        # compute self-attention
        if h is None:
            h = q

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        num_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shape_k = (self.n_heads, batch_size, graph_size, -1)
        shape_q = (self.n_heads, batch_size, num_query, -1)

        # Calculate queries
        Q = torch.matmul(qflat, self.W_query).view(shape_q)  # (n_heads, num_query, graph_size, key/val_size)
        # Calculate keys and values
        K = torch.matmul(hflat, self.W_key).view(shape_k)    # (n_heads, batch_size, graph_size, key/val_size)
        #V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility 
        compatibility_raw = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # (n_heads, batch_size, num_query, problem_size)
        compatibility = torch.tanh(compatibility_raw.mean(dim=0)) * 10.            # (batch_size, num_query, problem_size)
        return compatibility


class DenseToSparse(torch.nn.Module):
    """Convert from adj to edge_list while allowing gradients
    to flow through adj"""

    def forward(self, x, adj, weight):
        B = x.shape[0]
        N = x.shape[1]
        offset, row, col = torch.nonzero(adj > 0).t()
        edge_weight = weight[offset, row, col].float()
        row += offset * N
        col += offset * N
        edge_index = torch.stack([row, col], dim=0).long()
        x = x.view(B * N, x.shape[-1])
        batch_idx = (
            torch.arange(0, B, device=x.device).view(-1, 1).repeat(1, N).view(-1)
        )

        return x, edge_index, edge_weight, batch_idx


class NeuralTensorNetwork(torch.nn.Module):
    """
    Tensor Network module to calculate similarity vector.
    """
    def __init__(self, input_features, tensor_neurons):
        """
        :param args: Arguments object.
        """
        super(NeuralTensorNetwork, self).__init__()
        self.input_features = input_features
        self.tensor_neurons = tensor_neurons
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_features, self.input_features, self.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 2*self.input_features))
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.orthogonal_(self.weight_matrix)
        torch.nn.init.orthogonal_(self.weight_matrix_block)
        torch.nn.init.orthogonal_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = len(embedding_1)
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.input_features, -1))
        scoring = scoring.view(batch_size, self.input_features, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.input_features, 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = torch.relu(scoring + block_scoring + self.bias.view(-1))
        return scores


class Sinkhorn(nn.Module):
    """
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, log_forward=True, batched_operation=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        self.batched_operation = batched_operation # batched operation may cause instability in backward computation,
                                                   # but will boost computation.

    def forward(self, *input, dummy=False, **kwinput):
        if dummy:
            return self.forward_log_dummy(*input, **kwinput)
        else:
            return self.forward_log(*input, **kwinput)

    def forward_log_dummy(self, s, nrows=None, ncols=None, r=None, c=None, dummy_row=False, dtype=torch.float32):
        # computing sinkhorn with row/column normalization in the log space.
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        if r is None:
            log_r = torch.zeros(batch_size, s.shape[1], device=s.device)
        else:
            log_r = torch.log(r)
        if c is None:
            log_c = torch.zeros(batch_size, s.shape[2], device=s.device)
        else:
            log_c = torch.log(c)
        for b in range(batch_size):
            log_r[b, nrows[b]:] = -float('inf')
            log_c[b, ncols[b]:] = -float('inf')

        # operations are performed on log_s
        log_s = s / self.tau

        for b in range(batch_size):
            log_s[b, nrows[b]-1, ncols[b]-1] = -float('inf')

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            log_s = torch.cat((log_s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
            log_r = torch.cat((log_r, torch.full(dummy_shape[:2], -float('inf')).to(s.device)), dim=1)
            for b in range(batch_size):
                log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                log_s[b, nrows[b]:, :] = -float('inf')
                log_s[b, :, ncols[b]:] = -float('inf')

                log_r[b, ori_nrows[b]:nrows[b]] = 0

        if self.batched_operation:
            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    for b in range(batch_size):
                        log_sum[b, nrows[b]-1:, :] = 0
                    log_s = log_s - log_sum + log_r.unsqueeze(2)
                    log_s[torch.isnan(log_s)] = -float('inf')
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    for b in range(batch_size):
                        log_sum[b, :, ncols[b]-1:] = 0
                    log_s = log_s - log_sum + log_c.unsqueeze(1)
                    log_s[torch.isnan(log_s)] = -float('inf')

            if dummy_row and dummy_shape[1] > 0:
                log_s = log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            ret_log_s = torch.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s_b = log_s[b, row_slice, col_slice]
                log_r_b = log_r[b, row_slice]
                log_c_b = log_c[b, col_slice]

                for i in range(self.max_iter):
                    new_log_s_b = log_s_b.clone()
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
                        new_log_s_b[:-1, :] = log_s_b[:-1, :] - log_sum[:-1, :] + log_r_b[:-1].unsqueeze(1)
                    else:
                        log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
                        new_log_s_b[:, :-1] = log_s_b[:, :-1] - log_sum[:, :-1] + log_c_b[:-1].unsqueeze(0)

                    log_s_b = new_log_s_b

                ret_log_s[b, row_slice, col_slice] = log_s_b

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)

        # ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

        # for b in range(batch_size):
        #    row_slice = slice(0, nrows[b])
        #    col_slice = slice(0, ncols[b])
        #    log_s = s[b, row_slice, col_slice]

    def forward_log(self, s, nrows=None, ncols=None, r=None, c=None, dummy_row=False, dtype=torch.float32):
        # computing sinkhorn with row/column normalization in the log space.
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        if r is None:
            log_r = torch.zeros(batch_size, s.shape[1], device=s.device)
        else:
            log_r = torch.log(r)
        if c is None:
            log_c = torch.zeros(batch_size, s.shape[2], device=s.device)
        else:
            log_c = torch.log(c)
        for b in range(batch_size):
            log_r[b, nrows[b]:] = -float('inf')
            log_c[b, ncols[b]:] = -float('inf')

        # operations are performed on log_s
        log_s = s / self.tau

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            log_s = torch.cat((log_s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
            log_r = torch.cat((log_r, torch.full(dummy_shape[:2], -float('inf')).to(s.device)), dim=1)
            for b in range(batch_size):
                log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                log_s[b, nrows[b]:, :] = -float('inf')
                log_s[b, :, ncols[b]:] = -float('inf')

                log_r[b, ori_nrows[b]:nrows[b]] = 0

        if self.batched_operation:
            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    log_s = log_s - log_sum + log_r.unsqueeze(2)
                    log_s[torch.isnan(log_s)] = -float('inf')
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum + log_c.unsqueeze(1)
                    log_s[torch.isnan(log_s)] = -float('inf')

            if dummy_row and dummy_shape[1] > 0:
                log_s = log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            ret_log_s = torch.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s_b = log_s[b, row_slice, col_slice]
                log_r_b = log_r[b, row_slice]
                log_c_b = log_c[b, col_slice]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
                        log_s_b = log_s_b - log_sum + log_r_b.unsqueeze(1)
                    else:
                        log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
                        log_s_b = log_s_b - log_sum + log_c_b.unsqueeze(0)

                ret_log_s[b, row_slice, col_slice] = log_s_b

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)


class MultiplerNet(nn.Module):
    def __init__(self, state_dim):
        super(MultiplerNet, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        
    def forward(self, state):
        a = F.leaky_relu(self.l1(state))
        a = F.leaky_relu(self.l2(a))
        #return F.relu(self.l3(a))
        return F.softplus(self.l3(a)) # lagrangian multipliers can not be negative


if __name__ == '__main__':
    def test_mlp_net():
        mlp = MLPNet(100, 10)
        input = torch.rand(1, 100)
        out = mlp(input)
        print(out)

    def test_gcn_net():
        import networkx as nx

        a = nx.waxman_graph(n=100)
        print(dict(a.nodes(data=True)).values())
        
    test_gcn_net()