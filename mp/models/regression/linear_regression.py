# ------------------------------------------------------------------------------
# This class represents a linear regression model.
# ------------------------------------------------------------------------------

import torch.nn as nn
from mp.models.model import Model

class LinearRegression(Model):
    def __init__(
            self,
            input_shape=(32, 128, 128),
            output_shape=2,
            dimensions: int = 2,
            ):
        super().__init__(input_shape=input_shape, output_shape=output_shape)

    def forward(self):
        return nn.Linear(self.input_shape, self.output_shape)


class LinearRegression2D(LinearRegression):
    def __init__(self, *args, **kwargs):
        assert len(args[0]) == 3, "Input shape must have dimensions channels, width, height. Received: {}".format(args[0])
        predef_kwargs = {}
        predef_kwargs['dimensions'] = 2
        predef_kwargs.update(kwargs)
        super().__init__(*args, **predef_kwargs)

class LinearRegression3D(LinearRegression):
    def __init__(self, *args, **kwargs):
        assert len(args[0]) == 4, "Input shape must have dimensions channels, width, height, depth. Received: {}".format(args[0])
        predef_kwargs = {}
        predef_kwargs['dimensions'] = 3
        predef_kwargs.update(kwargs)
        super().__init__(*args, **predef_kwargs)
