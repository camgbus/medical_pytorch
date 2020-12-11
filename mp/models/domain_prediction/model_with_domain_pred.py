# ------------------------------------------------------------------------------
# Basic class for segmentation models.
# ------------------------------------------------------------------------------

from mp.models.model import Model
from itertools import chain


class FullModelWithDomainPred(Model):
    r"""
    TLDR: wrapper class to combine a model and a domain predictor + easy access to parameters groups
    This class assumes that a model at least contains an encoder part and a classifier part.
    For instance for a UNet the classifier will be the last convolutional layer.
    The encoding (product of the "encoder") can be the actual output of the UNet's encoder or the output of the decoder
    (depending on the paper)
    """
    def forward(self, x, detach=False):
        r"""Return a tuple: classification, domain prediction"""
        raise NotImplementedError

    def get_features_from_encoder(self, x):
        raise NotImplementedError

    def get_classification_from_features(self, x):
        raise NotImplementedError

    def get_domain_prediction_from_features(self, x):
        raise NotImplementedError

    def encoder_parameters(self):
        raise NotImplementedError

    def classifier_parameters(self):
        raise NotImplementedError

    def domain_predictor_parameters(self):
        raise NotImplementedError

    def parameters(self, **kwargs):
        return chain(self.encoder_parameters(), self.classifier_parameters(), self.domain_predictor_parameters())
