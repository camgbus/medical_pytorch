# ------------------------------------------------------------------------------
# This class represents different classification models.
# ------------------------------------------------------------------------------

import torch.nn as nn
from mp.models.model import Model

class CNN_Net3D(Model):   
    r"""This class represents a CNN for 3D image classification,
    detecting CT artefacts in CT slices."""
    def __init__(self, num_labels):
        super(CNN_Net3D, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a first 3D convolution layer
            nn.Conv3d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(4),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Defining a second 3D convolution layer
            nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Defining a third 3D convolution layer
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(kernel_size=2, stride=1)
        )

        self.linear_layers = nn.Sequential(
            # Output shape of cnn_layers
            nn.Linear(8 * 5 * 5 * 3, 128), 
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.50),
            nn.Linear(128, num_labels)
        )

    # Defining the forward pass    
    def forward(self, x):
        yhat = self.cnn_layers(x)
        #print(yhat.size())
        yhat = yhat.view(yhat.size(0), -1)
        #yhat = x.view(x.size(0), -1)
        yhat = self.linear_layers(yhat)
        return yhat