from mp.quantifiers.QualityQuantifier import SegImgQualityQuantifier
from mp.models.densities.density import Density_model
from mp.models.regression.dice_predictor import Dice_predictor
from mp.utils.feature_extractor import Feature_extractor
class IntBasedQuantifier(SegImgQualityQuantifier):

    def __init__(self, version='0.0'):
        super().__init__(version)

    def get_quality(self, mask, x=None):
        r"""Get quality values for a segmentation mask, optionally together with
        an image, according to one or more metrics. This method should be
        overwritten.

        Args:
            mask (numpy.Array): an int32 numpy array for a segmentation mask,
                with dimensions (channels, width, height, depth). The 'channels'
                dimension corresponds to the number of labels, the other
                dimensions should be the same as for x. All entries are 0 or 1.
            x (numpy.Array): a float64 numpy array for a 3D image, normalized so
                that all values are between 0 and 1. The array follows the
                dimensions (channels, width, height, depth), where channels == 1

        Returns (dict[str -> float]): a dictionary linking metric names to float
            quality estimates
        """
        
        #set features to use: 
        features=['density_distance','dice_scores','connected_components']

        # load density model
        density = Density_model(model='gaussian_kernel',add_to_name='')
        density.load_density()
        
        #load dice predictor
        dice_pred = Dice_predictor(features,add_to_name='')
        dice_pred.load()

        feature_extractor = Feature_extractor(density,features=features)
        
        features = feature_extractor.get_features(x,mask)

        dice_value = dice_pred.predict(features)

        return {'predicted dice score':dice_value}
        