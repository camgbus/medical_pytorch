from typing import Optional

import numpy as np
import torch.nn as nn

from mp.models.model import Model
from mp.models.segmentation.model_utils import ConvolutionalBlock, get_downsampling_layer


class DomainPredictor(Model):
    r"""The domain predictor is based on the UNet's implementation (but only its first half)"""
    def __init__(
            self,
            input_shape,
            nb_domains,
            out_channels_first_layer: int = 16,
            dimensions: int = 2,
            num_conv_blocks: int = 4,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            preactivation: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0.2
    ):
        super().__init__(input_shape, (nb_domains,))

        self.input_shape = input_shape

        self.conv_blocks = nn.ModuleList()
        self.dilation = initial_dilation

        # Convolutional blocks of encoder
        for idx in range(num_conv_blocks):
            conv_block = ConvolutionalBlock(
                dimensions,
                out_channels_first_layer,
                out_channels_first_layer,
                normalization=normalization,
                preactivation=preactivation,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )

            self.conv_blocks.append(conv_block)

            if self.dilation is not None:
                self.dilation *= 2

        # Projection block to reduce the nb of channels
        self.projector_block = ConvolutionalBlock(
            dimensions,
            out_channels_first_layer,
            1,  # That's the differences with the previous conv_blocks
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.dilation,
            dropout=dropout,
            kernel_size=1  # That's the differences with the previous conv_blocks
        )

        # Prep for classifier
        self.downsample = get_downsampling_layer(dimensions, pooling_type)
        self.classifier_input_size = int(np.prod([x >> 4 for x in self.input_shape[1:]]))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, self.classifier_input_size * 3 // 4),
            nn.ReLU(True),
            nn.Linear(self.classifier_input_size * 3 // 4, self.classifier_input_size // 4),
            nn.ReLU(True),
            nn.Dropout2d(p=dropout),
            nn.Linear(self.classifier_input_size // 4, nb_domains),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = self.downsample(x)

        x = self.projector_block(x)
        x = x.view(-1, self.classifier_input_size)

        domain_pred = self.classifier(x)
        return domain_pred


class DomainPredictor2D(DomainPredictor):
    def __init__(self, *args, **kwargs):
        assert len(args[0]) == 3, f"Input shape must have dimensions channels, width, height. Received: {args[0]}"
        predef_kwargs = {'dimensions': 2,
                         'num_conv_blocks': 4,
                         'normalization': 'batch',
                         'preactivation': True,
                         'padding': True}
        # Added this so there is no error between the skip connection and
        # feature mas shapes
        predef_kwargs.update(kwargs)
        super().__init__(*args, **predef_kwargs)


class DomainPredictor3D(DomainPredictor):
    def __init__(self, *args, **kwargs):
        assert len(args[0]) == 4, f"Input shape must have dimensions channels, width, height, depth. Received:{args[0]}"
        predef_kwargs = {'dimensions': 3,
                         'num_conv_blocks': 4,
                         'normalization': 'batch',
                         'padding': True}
        predef_kwargs.update(kwargs)
        super().__init__(*args, **predef_kwargs)
