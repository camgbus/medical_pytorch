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
        self.linear1 = nn.Linear(in_feat, 1000)
        self.linear2 = nn.Linear(1000, 10)
        self.linear3 = nn.Linear(10, out_feat)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(in_feat, 100)
        self.linear2 = nn.Linear(100, out_feat)
        #Loss huber
        #lr 0.0001
        #SGD
        
    def forward(self, x):
        # Reshape input based on batchsize
        inp_batch_size = list(x.size())[0]
        x = x.view(inp_batch_size, -1)
        yhat = self.linear1(x)
        yhat = self.relu(yhat)
        yhat = self.linear2(yhat)
        #yhat = self.relu(yhat)
        #yhat = self.linear3(yhat)
        #yhat = self.sigmoid(yhat)
        return yhat

class AlexNet(Model):
    r""" This class represents the AlexNet for image classification."""
    def __init__(self, num_labels):
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        classifier_input = self.alexnet.fc.in_features
        self.alexnet.fc = nn.Linear(classifier_input, num_labels)
        self.alexnet.eval()
        
    def forward(self, x):
        # Reshape input based on batchsize
        yhat = self.alexnet(x)
        return yhat