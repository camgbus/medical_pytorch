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

class EWC(object):
    r"""This class represents an object for the Elastic Weight Consolidation method."""
    def __init__(self, model, dataset, device):
        r"""Constructor. Model is a PyTorch model, and dataset represents a sample out of
            old scans on which the model has been trained in a previous task."""
        self.model = model
        self.dataset = dataset
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_metrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data)

    def _diag_fisher(self):
        precision_metrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_metrices[n] = Variable(p.data)

        self.model.eval()
        #for _, (x, y) in enumerate(self.dataset):
        for x in self.dataset:
            self.model.zero_grad()
            x = Variable(x)
            x = x.to(self.device)
            yhat = self.model(x).view(1, -1)
            label = yhat.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(yhat, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_metrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_metrices = {n: p for n, p in precision_metrices.items()}
        return precision_metrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_metrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss