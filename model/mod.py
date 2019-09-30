import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .mlp import BaseMLP
from .fit import BaseFit


class BaseMOD(nn.Module):
    def __init__(self, layer_sizes, num_net, l2_reg=1e-5):
        super(BaseMOD, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.layer_sizes = layer_sizes
        self.mlps = []
        for i in range(num_net):
            mlp = BaseMLP(layer_sizes, l2_reg)
            setattr(self, 'mlp'+str(i), mlp)
            self.mlps.append(mlp)

        self.double()
        self.to(self.device)
    
    def forward(self, x, dropout_rate=0, train=False, dropout_mask=None):
        out = torch.DoubleTensor(x).to(self.device) if isinstance(x, np.ndarray) else x
        out = torch.stack([mlp(out, dropout_rate, train, dropout_mask) for mlp in self.mlps]).mean(dim=0)
        return out if train else out.detach()

class MOD(BaseMOD, BaseFit):
    def __init__(self, layer_sizes, num_net, l2_reg=1e-5):
        super(MOD, self).__init__(layer_sizes, num_net)
        self.criterion = NLLLoss()
        self.optimizer = torch.optim.Adadelta(self.parameters(), weight_decay=l2_reg)
        
class NLLLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(NLLLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        means = input[:, [0]]
        stds = torch.abs(input[:, [1]])
        dist = torch.distributions.normal.Normal(means, stds)
        return -dist.log_prob(target).mean()