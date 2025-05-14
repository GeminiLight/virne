import torch
from torch import nn


class BarlowTwinsContrastiveLoss(nn.Module):

    def __init__(self, lambda_: float = None, batch_norm: bool = True, eps: float = 1e-5):
        super(BarlowTwinsContrastiveLoss, self).__init__()
        self.lambda_ = lambda_
        self.batch_norm = batch_norm
        self.eps = eps

    def forward(self, h1, h2) -> torch.FloatTensor:
        # import pdb; pdb.set_trace()
        assert h1.shape == h2.shape
        # if len(h1.shape) == 3:
        #     h1 = h1.reshape(-1, h1.shape[-1])
        #     h2 = h2.reshape(-1, h2.shape[-1])
        # if len(h1.shape) == 2:
        #     h1 = h1.reshape(-1, 64, 4)
        #     h2 = h2.reshape(-1, 64, 4)

        batch_size = h1.size(0)
        feature_dim = h1.size(1)

        if self.lambda_ is None:
            lambda_ = 1. / feature_dim
        else:
            lambda_ = self.lambda_

        if self.batch_norm:
            z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + self.eps)
            z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + self.eps)
            c = torch.matmul(z1_norm.transpose(-2, -1), z2_norm) / batch_size
        else:
            c = torch.matmul(h1.transpose(-2, -1), h2) / batch_size

        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        loss = (1 - c.diagonal()).pow(2).sum()
        loss += lambda_ * c[off_diagonal_mask].pow(2).sum()
        return loss.mean()
