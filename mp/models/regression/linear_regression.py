# ------------------------------------------------------------------------------
# This class represents a linear regression model.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
from mp.models.model import Model
import torchvision.models as models

class LinearRegression(Model):
    r""" This class represents a simple cnn Model with a regression head."""
    def __init__(self, out_feat):
        super(LinearRegression, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a first 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining a second 2D convolution layer
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining a third 2D convolution layer
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining a forth 2D convolution layer
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.regression_layer = nn.Sequential(
            # Output shape of cnn_layers
            #nn.Linear(8 * 18 * 18, out_feat),
            #nn.Sigmoid()
            nn.Linear(8 * 18 * 18, 1),
            #nn.ReLU(inplace=True),
            #nn.Linear(5, out_feat),
            #nn.Sigmoid()
        )

    # Defining the forward pass    
    def forward(self, x):
        yhat = self.cnn_layers(x)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.regression_layer(yhat)
        return yhat

class LinearRegressionOrig(Model):
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