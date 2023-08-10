import torch
import torch.nn as nn
import torch.nn.functional as F
from virne.solver.learning.neural_network import *


class ActorCriticBase(nn.Module):

    def __init__(self, ):
        super(ActorCriticBase, self).__init__()
    
    def act(self, x):
        return self.actor(x)
    
    def evaluate(self, x):
        if not hasattr(self, 'critic'):
            return None
        return self.critic(x)