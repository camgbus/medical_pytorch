import torch

from mp.eval.losses.loss_abstract import LossAbstract


class ConfusionLoss(LossAbstract):
    def forward(self, x, target):
        # We only care about x
        log = torch.log(x)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum, x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss
