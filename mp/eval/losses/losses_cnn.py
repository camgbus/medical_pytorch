# ------------------------------------------------------------------------------
# Collection of loss metrics that can be used during training, including MAE,
# MSE and Huber Loss.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import functional as F
from torch.autograd import Variable
from mp.eval.losses.loss_abstract import LossAbstract

class LossCEL(LossAbstract):
    r"""Cross Entropy loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.cel = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target):
        return self.cel(output, target)

class LossNLL(LossAbstract):
    r"""Negative Log Likelihood loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.nll = nn.NLLLoss(reduction='mean')

    def forward(self, output, target):
        return self.nll(output, target)