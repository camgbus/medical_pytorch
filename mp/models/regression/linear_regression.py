# ------------------------------------------------------------------------------
# This class represents a linear regression model.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
from mp.models.model import Model
import torchvision.models as models

class LinearRegression(Model):
    r""" This class represents a simple linear regression Model."""
    def __init__(self, in_feat, out_feat):
        super(LinearRegression, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(in_feat, 1048)
        self.linear2 = nn.Linear(1048, out_feat)
        
    def forward(self, x):
        # Reshape input based on batchsize
        inp_batch_size = list(x.size())[0]
        x = x.view(inp_batch_size, -1)
        yhat = self.linear1(x)
        yhat = self.relu(yhat)
        yhat = self.dropout(yhat)
        yhat = self.linear2(yhat)
        yhat = self.sigmoid(yhat)
        return yhat