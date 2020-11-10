# ------------------------------------------------------------------------------
# This class represents a linear regression model.
# ------------------------------------------------------------------------------

import torch.nn as nn
from mp.models.model import Model

class LinearRegression(Model):
    r""" This class represents a simple linear regression Model."""
    def __init__(self, in_feat, out_feat, batch_size):
        super(LinearRegression, self).__init__()
        self.batch_size = batch_size
        self.linear = nn.Linear(in_feat, out_feat)
        
    def forward(self, x):
        # Reshape input based on batchsize
        x = x.contiguous()
        x = x.view(self.batch_size, -1)
        yhat = self.linear(x)
        return yhat