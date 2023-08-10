import torch.nn as nn
import torch.nn.functional as F


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
