import os
import numpy as np
from mp.models.densities.density import Density_model
from mp.utils.intensities import get_intensities, load_intensities
from mp.models.regression.dice_predictor import Dice_predictor
from mp.utils.feature_extractor import Feature_extractor
from mp.utils.preprocess_utility_functions import basic_preprocessing
from mp.utils.preprocess_utility_functions import extract_features_all_data,compute_all_prediction_dice_scores

def train_int_based_quantifier(preprocess=True,train_dice_pred=True,verbose=False,label=1):
    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        if preprocess:
            if verbose:
                print('Preprocessing data ...')
            basic_preprocessing(label=label) 
        if train_dice_pred:
            if verbose:
                print('Training dice predictor ...')
            extract_features_all_data(label=label)
            compute_all_prediction_dice_scores()
            train_dice_predictor(verbose=verbose,label=label)
    else : 
        print('This is only for Train mode')

def train_dice_predictor(model_name='standart',feature_extractor=None,data_describtion = 'all of train data',
                            model_describtion = 'SVR',verbose=False,label=1,**kwargs):
    '''Trains a dice predictor model based on features extracted from image-seg pairs

    Args:
        model_name (str): The name of the model
        feature_extractor (instance of Feature_extractor class):The names of the features to be used
        data_describtion (str): a describtion of the used data
        model_describtion (str): a describtion of the used model and density model
        verbose(bool): whether the model shall print output
    '''
    if not feature_extractor:
        feature_extractor = Feature_extractor()

    #initiate model
    dice_pred = Dice_predictor(features=feature_extractor.features,version=model_name,verbose=verbose,label=label)

    #Load the features 
    X_train,y_train = feature_extractor.collect_train_data()

    #train model
    dice_pred.train(X_train,y_train, data_describtion, model_describtion,**kwargs)
    
    if verbose:
        dice_pred.print_description()

# # unused, deprecated  
# def train_density(model = '',
#                     ending_of_model = '',
#                     list_of_paths = [os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])],
#                     data_mode='JIP',
#                     data_describtion = 'training on nearly all data from COVID-RACOON at 5.4.2021', 
#                     model_describtion = 'gaussian model with bw 0.005', 
#                     precom_intensities=[], 
#                     verbose=False, 
#                     label = 1,
#                     **kwargs):
#     '''Trains a density model from a list of given paths to directories, where img-seg pairs can be found
#         and stores it
#     Args:
#         model (str): The name of the model being trained. Gives the first part of the name
#         ending_of_model (str): in order to specify model with name 'model' shall be trained ;
#             is the second and last part of the density models name
#         list_of_paths (list(str)): Every path in this path leads to a directory, that is then later 
#             iteratet over in order to gather intensity values
#         data_mode (str): the mode in which the data is stored, for further insight, look at Iterators.py
#         data_describtion (str): a string to describe the data used for training (e.g. which instances from
#             which dataset, when the training was done, etc)
#         model_describtion (str): a string to describe the specifications of the density model (e.g. bandwith, 
#             other parameters, etc)
#         precom_intensities (list(str)): a list of name of already computed intensity values, that are loaded 
#             from op_pers_dirs/intensities, dont need .npy ending
#         verbose (bool): If consolte outputs should be printed
#         **kwargs is used for the specifics of the model to be trained
#     '''

#     #initialise density model
#     if model and ending_of_model:
#         density_model = Density_model(model=model,version=ending_of_model,verbose=verbose,label=label)
#     if model and not ending_of_model:
#         density_model = Density_model(model=model,verbose=verbose,label=label)
#     if not model and ending_of_model:
#         density_model = Density_model(version=ending_of_model,verbose=verbose,label=label)
#     if not model and not ending_of_model:
#         density_model = Density_model(verbose=verbose,label=label)
    

#     #get the intensity values from the images 
#     if verbose:
#         print('Getting intensity values')
#     intensity_values = get_intensities(list_of_paths,mode=data_mode)

#     #load already computed intensities and merge the two
#     pre_intensities = load_intensities(precom_intensities)
#     intensity_values = np.append(intensity_values,pre_intensities)


#     #train density model
#     if verbose:
#         print('Training density model')
#     density_model.train_density(intensity_values,
#                                 data_describtion,model_describtion,**kwargs)

#     if verbose:
#         density_model.plot_density()



