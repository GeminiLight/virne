import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, PNAConv, NNConv, SAGEConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from .graph_conv import EdgeFusionGATConv


def get_gnn_class(gnn_type):
    if gnn_type == 'gcn':
        return GCNConvNet
    elif gnn_type == 'gat':
        return GATConvNet
    elif gnn_type == 'edge_fusion_gat':
        return EdgeFusionGATConvNet
    elif gnn_type == 'deep_edge_gat':
        return DeepEdgeFeatureGAT
    else:
        raise NotImplementedError(f'GNN type {gnn_type} is not supported.')


def to_sparse_batch(node_dense_embeddings, mask):
    embedding_dim = node_dense_embeddings.shape[-1]
    mask = mask.unsqueeze(-1).expand_as(node_dense_embeddings)
    node_sparse_embeddings  = torch.masked_select(node_dense_embeddings, mask).reshape(-1, embedding_dim)
    return node_sparse_embeddings


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

        # self._init_parameters()

    def get_conv(self, input_dim, output_dim, edge_dim=None, aggr='add', bias=True, **kwargs):
        raise NotImplementedError

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
        # self._init_parameters()

    # def _init_parameters(self):
    #     for layer_id in list(range(self.num_mid_layers)) + ['s', 'e']:
    #         nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_src.weight)
    #         nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_dst.weight)
    #         if self.edge_dim is not None:
    #             nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin_edge.weight)
    #         if layer_id not in ['s', 'e']:
    #             nn.init.orthogonal_(getattr(self, f'weight_{layer_id}'))

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
            raise NotImplementedError

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
