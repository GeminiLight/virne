from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList, Sequential, ReLU


from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class NNConv(MessagePassing):
    """The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, edge_dim: Optional[int] = None, aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if edge_dim is not None:
            self.edge_nn = Linear(in_channels[1], out_channels, bias=False,
                              weight_initializer='uniform')
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_weight:
            self.lin = Linear(in_channels[1], out_channels, bias=False,
                              weight_initializer='uniform')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.pre_nns = ModuleList()
        modules = [Linear(3 * self.F_in, self.F_in)]
        for _ in range(1):
            modules += [ReLU()]
            modules += [Linear(self.F_in, self.F_in)]
        self.pre_nns.append(Sequential(*modules))


        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        if self.root_weight:
            self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root_weight:
            out += self.lin(x_r)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)
        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr}, nn={self.nn})')


from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (Adj, NoneType, OptPairTensor, OptTensor,
                                    Size)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class EdgeFusionGATConv(MessagePassing):
    """The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(EdgeFusionGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.F_in = out_channels

        self.pre_nns = ModuleList()
        modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
        for _ in range(1):
            modules += [ReLU()]
            modules += [Linear(self.F_in, self.F_in)]
        self.pre_nns.append(Sequential(*modules))

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        # self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        # self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            # self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
            self.register_parameter('att_edge', None)
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        # glorot(self.att_src)
        # glorot(self.att_dst)
        # glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        """
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        # alpha_src = (x_src * self.att_src).sum(dim=-1)
        # alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        # alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")


        alpha = self._alpha = None

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=None, edge_attr=edge_attr,
                             size=size)

        # alpha = self._alpha
        # assert alpha is not None
        # self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        # alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        # if edge_attr is not None:
        #     if edge_attr.dim() == 1:
        #         edge_attr = edge_attr.view(-1, 1)
        #     assert self.lin_edge is not None
        #     raw_edge_attr = self.lin_edge(edge_attr)
        #     edge_attr = raw_edge_attr.view(-1, self.heads, self.out_channels)
        #     alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
        #     alpha = alpha + alpha_edge

        # alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, index, ptr, size_i)
        # self._alpha = alpha  # Save for later use.
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_embedding = self.lin_edge(edge_attr)
            h_edge_attr = edge_embedding.view(-1, 1, self.F_in)
            h = torch.cat([x_i, x_j, h_edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        new_x_j = torch.stack(hs, dim=1)
        # new_x_j * alpha.unsqueeze(-1) * alpha.unsqueeze(-1)
        return new_x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
