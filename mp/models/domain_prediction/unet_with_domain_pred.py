# ------------------------------------------------------------------------------
# Basic class for segmentation models.
# ------------------------------------------------------------------------------

from itertools import chain

from mp.models.domain_prediction.model_with_domain_pred import FullModelWithDomainPred
from mp.models.model import Model
from mp.models.segmentation.unet_fepegar import UNet


class UNetWithDomainPred(FullModelWithDomainPred):
    r"""
    TLDR: A UNet but its encoder + bottleneck + decoder parts act as the "encoder"
    The UNet's classifier is the classifier.
    A domain predictor is attached after the UNet's decoder.
    Architecture from this paper: https://www.biorxiv.org/content/10.1101/2020.10.09.332973v1.full.pdf+html
    Implementation @ https://github.com/nkdinsdale/Unlearning_for_MRI_harmonisation
    """

    def __init__(self, unet: UNet, domain_predictor: Model, input_shape=(1, 32, 32), output_shape=2):
        super().__init__(input_shape, output_shape=output_shape)
        self.unet = unet
        self.domain_predictor = domain_predictor

    def forward(self, x):
        features = self.get_features_from_encoder(x)
        return self.get_classification_from_features(features), \
               self.get_domain_prediction_from_features(features)

    def get_features_from_encoder(self, x):
        return self.unet.encode(x)

    def get_classification_from_features(self, x):
        return self.unet.classify(x)

    def get_domain_prediction_from_features(self, x):
        return self.domain_predictor(x)

    def encoder_parameters(self):
        return chain(self.unet.encoder.parameters(),
                     self.unet.bottom_block.parameters(),
                     self.unet.decoder.parameters(),
                     )

    def classifier_parameters(self):
        return self.unet.classifier.parameters()

    def domain_predictor_parameters(self):
        return self.domain_predictor.parameters()
