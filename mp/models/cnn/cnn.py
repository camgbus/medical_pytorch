# ------------------------------------------------------------------------------
# This class represents different classification models.
# ------------------------------------------------------------------------------

import torch.nn as nn
from mp.models.model import Model

class CNN_Net3D(Model):   
    r"""This class represents a CNN for 3D image classification,
    detecting CT artefacts in CT slices.
    The input image needs to have the size 299x299x10. Otherwise the
    number of input features for the Linear layer needs to be changed!"""
    def __init__(self, num_labels):
        super(CNN_Net3D, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a first 3D convolution layer
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Defining a second 3D convolution layer
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Defining a third 3D convolution layer
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
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
            nn.Linear(8 * 4 * 4 * 3, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_labels)
        )

    # Defining the forward pass    
    def forward(self, x):
        yhat = self.cnn_layers(x)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.linear_layers(yhat)
        return yhat

# Create a CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.conv_layer3 = self._conv_layer_set(64, 128)
        self.fc1 = nn.Linear(128 * 6 * 6 * 5, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch1 = nn.BatchNorm1d(1024)
        self.batch2 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)        
        
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
        yhat = self.conv_layer3(yhat)
        yhat = yhat.view(yhat.size(0), -1)
        yhat = self.fc1(yhat)
        yhat = self.relu(yhat)
        yhat = self.batch1(yhat)
        yhat = self.drop(yhat)
        yhat = self.fc2(yhat)
        yhat = self.relu(yhat)
        yhat = self.batch2(yhat)
        yhat = self.drop(yhat)
        yhat = self.fc3(yhat)
        return yhat