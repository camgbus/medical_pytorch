# ------------------------------------------------------------------------------
# A standard regression agent, which performs softmax in the outputs.
# ------------------------------------------------------------------------------

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax

class RegressionAgent(Agent):
    r"""An Agent for regression models."""
    def __init__(self, *args, **kwargs):
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['Mean_Squared_Error']
        super().__init__(*args, **kwargs)

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        outputs = softmax(outputs)
        return outputs