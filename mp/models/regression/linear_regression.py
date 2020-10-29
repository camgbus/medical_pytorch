# ------------------------------------------------------------------------------
# This class represents a linear regression model.
# ------------------------------------------------------------------------------

import torch.nn as nn
from mp.models.model import Model

class LinearRegression(Model):
    r""" This class represents a simple linear regression Model."""
    def __init__(self, in_feat, out_feat):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        
    def forward(self,x):
        yhat = self.linear(x)
        return yhat