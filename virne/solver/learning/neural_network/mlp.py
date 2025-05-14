import torch.nn as nn


class MLPNet(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers=2, embedding_dims=None, batch_norm=False, dropout_prob=1.0):
        super(MLPNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        if isinstance(embedding_dims, int):
            embedding_dims = [embedding_dims] * (num_layers-1)
        elif isinstance(embedding_dims, list): 
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