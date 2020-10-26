from mp.models.model import Model
import torch


class IRMGamesModel(Model):
    r"""An ensembling model for IRM Games (Invariant Risk Minimization Games, Ahuja et al.)
    found at https://arxiv.org/pdf/2002.04692.pdf
    Code inspired by fork: https://github.com/Bertinus/IRM-games/

    Args:
        models (list): a list of models (one for each data generation environment)
        representation_learner (Model): TODO add representation learner to IRMGamesModel
                                        an optional Model used to learn a new data representation
                                        (before feeding the data to the models)
        input_shape tuple (int): Input shape with the form
            (channels, width, height, Opt(depth))
        output_shape (Obj): output shape, which takes different forms depending
            on the problem
    """
    def __init__(self, models, input_shape=(1, 32, 32), output_shape=2, representation_learner=None):
        super().__init__(input_shape=input_shape, output_shape=output_shape)
        self.models = models
        self.representation_learner = representation_learner
        self.n_env = len(self.models)
        assert self.n_env > 0, "At least one model needs to be supplied"

    def forward(self, x):
        self.predict(x, keep_grad_idx=None)

    def predict(self, x, keep_grad_idx=None):
        r"""
        This method exists for model training purposes.
        We need to be able to compute the gradient for only one model at a time.

        Args:
            x (torch.Tensor): the input
            keep_grad_idx (optional, int): the index of the model for which the gradient needs to be computed
        """

        y = None
        for idx, model in enumerate(self.models):
            if idx == keep_grad_idx:
                y_env = model(x)
            else:
                with torch.no_grad():
                    y_env = model(x)

            # FIXME find elegant way of finding out y's shape
            if y is None:
                y = y_env
            else:
                y += y_env
        y /= self.n_env

        return y

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Is this necessary?
        for model in self.models:
            model.to(*args, **kwargs)
