# ------------------------------------------------------------------------------
# This class represents different classification models.
# ------------------------------------------------------------------------------

import torch.nn as nn
from mp.models.model import Model

class CNN_Net2D(Model):   
    r"""This class represents a CNN for 2D image classification,
    detecting CT artefacts in CT slices.
    The input image needs to have the size 299x299. Otherwise the
    number of input features for the Linear layer needs to be changed!"""
    def __init__(self, num_labels):
        super(CNN_Net2D, self).__init__()
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

        self.linear_layers = nn.Sequential(
            # Output shape of cnn_layers
            nn.Linear(8 * 18 * 18, num_labels)
        )

    # Defining the forward pass    
    def forward(self, x):
        yhat = self.cnn_layers(x)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.linear_layers(yhat)
        return yhat

class CNN_Net3D(Model):   
    r"""This class represents a CNN for 3D image classification,
    detecting CT artefacts in CT slices.
    The input image needs to have the size 299x299x10. Otherwise the
    number of input features for the Linear layer needs to be changed!"""
    def __init__(self, num_labels):
        super(CNN_Net3D, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a first 3D convolution layer
            nn.Conv3d(10, 14, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(14),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Defining a second 3D convolution layer
            nn.Conv3d(14, 18, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(18),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Defining a third 3D convolution layer
            nn.Conv3d(18, 18, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(18),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Defining a forth 3D convolution layer
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            # Output shape of cnn_layers
            nn.Linear(8 * 18 * 18, num_labels)
        )

    # Defining the forward pass    
    def forward(self, x):
        yhat = self.cnn_layers(x)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.linear_layers(yhat)
        return yhat

# Create a CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layer1 = self._conv_layer_set(10, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(2**10*64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        # Set 1
        yhat = self.conv_layer1(x)
        yhat = self.conv_layer2(yhat)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.fc1(yhat)
        yhat = self.relu(yhat)
        yhat = self.batch(yhat)
        yhat = self.drop(yhat)
        yhat = self.fc2(yhat)
        return yhat