# ------------------------------------------------------------------------------
# This class represents different classification models.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
from mp.models.model import Model
import torchvision.models as models

class AlexNet(Model):
    r"""This class represents the AlexNet for image classification."""
    def __init__(self, num_labels):
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        classifier_input = self.alexnet.classifier[-1].in_features
        self.alexnet.classifier[-1] = nn.Linear(classifier_input, num_labels)
        self.alexnet.eval()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Reshape input based on batchsize
        yhat = self.alexnet(x)
        yhat = self.sigmoid(yhat)
        return yhat

class CNN_Net2D(Model):   
    r"""This class represents a CNN for 2D image classification.
    The input image needs to have the size 299x299. Otherwise the
    number of input features for the Linear layer need to be changed!"""
    def __init__(self, num_labels):
        super(CNN_Net2D, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a first 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining a second 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 74 * 74, num_labels)
        )

    # Defining the forward pass    
    def forward(self, x):
        yhat = self.cnn_layers(x)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.linear_layers(yhat)
        return yhat