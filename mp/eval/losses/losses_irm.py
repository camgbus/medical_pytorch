import torch

from .loss_abstract import LossAbstract


class IRMLossAbstract(LossAbstract):
    r"""
    A loss for IRM. These losses often compute a ERM term (a erm call each) and
    a penalty for each environment (a forward call each)
    and then combine the list of ERM and penalty terms into a final loss (finalize_loss call).
    """

    def __init__(self, erm_loss, device="cuda:0"):
        super().__init__(device=device)
        self.erm_loss = erm_loss
        # Could also have been passed as arg to finalize_loss, but might be needed for eval dict
        self.penalty_weight = 1.

    def forward(self, output, target):
        raise NotImplementedError

    def erm(self, output, target):
        return self.erm_loss(output, target)

    def finalize_loss(self, nlls, penalties):
        r"""
        Computes the overall loss for the batch across environments.
        Args:
            nlls (list): risk for each environment
            penalties (list): penalty for each environment
        """
        raise NotImplementedError

    def get_evaluation_dict(self, output, target):
        eval_dict = super().get_evaluation_dict(output, target)
        # Just need to add the ERM term
        eval_dict.update(self.erm_loss.get_evaluation_dict(output, target))
        return eval_dict


class IRMv1Loss(IRMLossAbstract):
    r"""
    IRMv1 loss from Invariant Risk Minimization, M. Arjovsky et al.
    https://arxiv.org/pdf/1907.02893.pdf
    """

    def forward(self, output, target):
        scale = torch.tensor(1.).requires_grad_()
        loss = self.erm_loss(output * scale, target)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def finalize_loss(self, nlls, penalties):
        loss = torch.stack(nlls, dim=0).mean() + self.penalty_weight * torch.stack(penalties, dim=0).mean()
        if self.penalty_weight > 1.:
            loss /= self.penalty_weight
        return loss


class VRexLoss(IRMLossAbstract):
    r"""
    V-REx loss from Out-of-Distribution Generalization via Risk Extrapolation, D. Krueger et al.
    https://arxiv.org/pdf/2003.00688.pdf
    """

    def forward(self, output, target):
        return torch.tensor(0., device=self.device, dtype=output.dtype)

    def finalize_loss(self, nlls, penalties):
        loss = torch.stack(nlls, dim=0).sum() + self.penalty_weight * torch.stack(nlls, dim=0).var()
        if self.penalty_weight > 1.:
            loss /= self.penalty_weight
        return loss


class MMRexLoss(IRMLossAbstract):
    r"""
    MM-REx loss from Out-of-Distribution Generalization via Risk Extrapolation, D. Krueger et al.
    https://arxiv.org/pdf/2003.00688.pdf
    """

    def forward(self, output, target):
        return torch.tensor(0., device=self.device, dtype=output.dtype)

    def finalize_loss(self, nlls, penalties):
        nlls_tensor = torch.stack(nlls, dim=0)
        if self.penalty_weight > 1.:
            loss = (1. / self.penalty_weight + len(nlls)) * torch.max(nlls_tensor) - nlls_tensor.sum()
        else:
            loss = (1. + len(nlls) * self.penalty_weight) * torch.max(nlls_tensor) \
                   - self.penalty_weight * nlls_tensor.sum()

        return loss


class ERMWrapper(IRMLossAbstract):
    r"""
    Wrapper for AbstractLoss
    """

    def forward(self, output, target):
        return torch.tensor(0., device=self.device, dtype=output.dtype)

    def finalize_loss(self, nlls, penalties):
        return torch.stack(nlls, dim=0).sum()

    def get_evaluation_dict(self, output, target):
        return self.erm_loss.get_evaluation_dict(output, target)
