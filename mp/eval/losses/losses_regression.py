# ------------------------------------------------------------------------------
# Collection of loss metrics that can be used during training, including MAE,
# MSE and Huber Loss.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from mp.eval.losses.loss_abstract import LossAbstract

class LossMAE(LossAbstract):
    r"""Mean absolute error loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.mae = nn.L1Loss(reduction='mean')

    def forward(self, output, target):
        return self.mae(output, target)

class LossMSE(LossAbstract):
    r"""Mean squared error loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        return self.mse(output, target)

class LossHuber(LossAbstract):
    r"""Huber loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.huber = nn.SmoothL1Loss(reduction='mean')

    def forward(self, output, target):
        return self.huber(output, target)

class LossCEL(LossAbstract):
    r"""Cross Entropy loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.cel = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target):
        return self.cel(output, target)