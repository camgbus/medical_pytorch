# ------------------------------------------------------------------------------
# This class represents a linear regression model.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
from mp.models.model import Model

class LinearRegression(Model):
    r""" This class represents a simple linear regression Model."""
    def __init__(self, in_feat, out_feat):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(in_feat, 1000)
        self.linear2 = nn.Linear(1000, 10)
        self.linear3 = nn.Linear(10, out_feat)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape input based on batchsize
        inp_batch_size = list(x.size())[0]
        x = x.view(inp_batch_size, -1)
        yhat = self.linear1(x)
        yhat = self.relu(yhat)
        yhat = self.linear2(yhat)
        yhat = self.relu(yhat)
        yhat = self.linear3(yhat)
        return yhat