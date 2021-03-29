import numpy as np
from mp.quantifiers.QualityQuantifier import SegImgQualityQuantifier
from mp.models.densities.density import Density_model
from mp.models.regression.dice_predictor import Dice_predictor
from mp.utils.feature_extractor import Feature_extractor
import torch 
import torchio

class IntBasedQuantifier(SegImgQualityQuantifier):

    def __init__(self, version='0.0'):
        super().__init__(version)

    def get_quality(self, mask, x=None):
        r"""Get quality values for a segmentation mask, optionally together with
        an image, according to one or more metrics. This method should be
        overwritten.

        Args:
            New Version: mask and x are lists of paths to images/segmentations

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
        # 1. prepare for use and load models 

        #set features to use: 
        features=['density_distance','dice_scores','connected_components']

        # load density model
        density = Density_model(model='gaussian_kernel',add_to_name='dummy')
        density.load_density()
        
        #load dice predictor
        dice_pred = Dice_predictor(features,add_to_name='UK_Fra_dummy')
        dice_pred.load()

        #load feature extractor
        feature_extractor = Feature_extractor(density,features=features)
        
        #get score
        score = self.get_score_from_paths(x,mask,density,dice_pred,feature_extractor)

        return {'predicted dice score':score}


    def get_score_from_paths(self,list_img_paths,list_seg_paths,density,dice_pred,feature_extractor):
        '''gets a list paths to images and a list of paths to segmentations and computes the average 
                score over these img-seg pairs.

        Args:
            list_img_paths (list(str)):the list of paths to the images
            list_seg_paths (list(str)):the list of paths to the segmentations
            density (instance of Density_model): A density model used to compute score 
            dice_pred (instance of Dice_predictor): A model that tries to predict the dice score
            feature extractor (instance of Feature_extractor): a util instance, that computes the 
                features from a given img-seg pair
        
        Returns (float): the predicted score averaged over all the images
        '''
        list_of_scores = []
        for img_path,seg_path in zip(list_img_paths,list_seg_paths):
            img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
            seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0] 
            features = feature_extractor.get_features(img,seg)
            dice_value = dice_pred.predict(features)
            list_of_scores.append(dice_value)
        
        # after scores are computed, we can compute the mean and return it
        arr_of_scores = np.array(list_of_scores)
        score = np.mean(arr_of_scores)
        return score


        