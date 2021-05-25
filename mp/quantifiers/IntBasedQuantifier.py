import numpy as np
from mp.quantifiers.QualityQuantifier import SegImgQualityQuantifier
from mp.models.densities.density import Density_model
from mp.models.regression.dice_predictor import Dice_predictor
from mp.utils.feature_extractor import Feature_extractor
import torch 
import torchio
from mp.utils.preprocess_utility_functions import basic_preprocessing, extract_features_all_data
import os
import json 

class IntBasedQuantifier(SegImgQualityQuantifier):

    def __init__(self, label=1, version='0.0'):
        super().__init__(version)

        self.label = label

        self.feat_extr = Feature_extractor()
        self.dice_pred = Dice_predictor(label=label,version='standart')
        self.dice_pred.load()

        self.work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"])
        self.output_path = os.path.join(os.environ["WORKFLOW_DIR"],os.environ["OPERATOR_OUT_DIR"],'segmentation_quality_metrics.json')

    def preprocess_data(self):
        '''copies the data from the input dir into preprocessed scaled dir 
        and extracts the features of the data'''
        os.environ["INFERENCE_OR_TRAIN"] = 'inference'
        basic_preprocessing(label=self.label)
        extract_features_all_data(label=self.label)

    def get_quality(self):
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
        self.preprocess_data()
        output_dict=dict()
        for id in os.listdir(self.work_path):
            id_dict = dict()
            path_to_features = os.path.join(self.work_path,id,'seg','features.json')
            feature_vec = self.feat_extr.read_feature_vector(path_to_features)
            prediction = self.dice_pred.predict([feature_vec])[0]
            id_dict['dice_pred']=prediction
            output_dict[id]=id_dict
        
        with open(self.output_path,'w') as file:
            json.dump(output_dict,file)


        