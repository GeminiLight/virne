import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from .auto_encoder import InnerProductDecoder, ARGVA


class GraphViNEConv(MessagePassing):
    
    def __init__(self, in_channels, out_channels, normalize=False, bias=True,
                 **kwargs):
        super(GraphViNEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""

        if torch.is_tensor(x):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_rel(out) + self.lin_root(x[1])

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def message(self, x_j, edge_weight):
        # print(f'Message is {x_j} with weight {}')
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Discriminator(torch.nn.Module):

    def __init__(self, in_channel, hidden, out_channel):
        super(Discriminator, self).__init__()
        self.dense1 = torch.nn.Linear(in_channel, hidden)
        self.dense2 = torch.nn.Linear(hidden, out_channel)
        self.dense3 = torch.nn.Linear(out_channel, out_channel)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Encoder(torch.nn.Module):

    def __init__(self, in_channels, hidden, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GraphViNEConv(in_channels, hidden)
        self.conv_mu = GraphViNEConv(hidden, out_channels)
        self.conv_logvar = GraphViNEConv(hidden, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        return self.conv_mu(x, edge_index, edge_attr), self.conv_logvar(x, edge_index, edge_attr)


class GraphVineDecoder(torch.nn.Module):

    def __init__(self, input_numbers, hidden_depth, link_creator_numbers, output_numbers):
      super(GraphVineDecoder, self).__init__()
      self.dense1 = torch.nn.Linear(input_numbers, hidden_depth)
      self.dense2 = torch.nn.Linear(hidden_depth, output_numbers)
      self.dense3 = torch.nn.Linear(hidden_depth, link_creator_numbers)
      self.inner_product = InnerProductDecoder()

    def forward(self, z, edge_index, sigmoid=True):
      z = F.relu(self.dense1(z))
      zprim = self.dense2(z)
      aprim = self.dense3(z)
      
      return zprim, self.inner_product(aprim, edge_index)