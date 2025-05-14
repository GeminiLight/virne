from itertools import chain
import torch.nn as nn
import torch.nn.functional as F


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