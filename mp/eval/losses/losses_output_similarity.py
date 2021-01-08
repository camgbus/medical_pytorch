# ------------------------------------------------------------------------------
# Cosine similarity and distance losses.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from mp.eval.losses.loss_abstract import LossAbstractOutputs

class LossCosineSimilarity(LossAbstractOutputs):
    r"""Cosine similarity loss (the similarity should be reduced)."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.cs = nn.CosineSimilarity(dim=0)

    def forward(self, x1, x2):
        r"""x1 and x2 are two model outputs. All elements should be between 0
        and 1."""
        x1 = torch.flatten(x1.float(), start_dim=0)
        x2 = torch.flatten(x2.float(), start_dim=0)
        return self.cs(x1, x2)

class LossCosineDistance(LossAbstractOutputs):
    r"""Inverse of cosine similarity."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.cs = LossCosineSimilarity()

    def forward(self, x1, x2):
        r"""x1 and x2 are two model outputs. All elements should be between 0
        and 1."""
        return 1 - self.cs(x1, x2)
