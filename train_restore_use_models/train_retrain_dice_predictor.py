from mp.models.regression.dice_predictor import Dice_predictor
from mp.utils.feature_extractor import Feature_extractor
from mp.models.densities.density import Density_model
import numpy as np

def train_dice_predictor(model_name='standart',feature_extractor=Feature_extractor(),data_describtion = 'all of train data',
                            model_describtion = 'MLP',verbose=False,**kwargs):
    '''Trains a dice predictor model based on features extracted from image-seg pairs
     a standart setting for kwargs is: 
     kwargs = {'solver'='adam','lr'='adaptive','hidden_layer_sizes'=(10,30,50,50,20)}

    Args:
        model_name (str): The name of the model
        feature_names (list(str)):The names of the features to be used
        dens_model (str): The first part of the density model name 
        dens_add_name (str): the second part of the density model name 
            specifying the exact model 
        list_of_paths (list(str)): a list of paths to the directories, where features
            shall be extracted from 
        names_extracted_features (list(str)): names of already extracted features, that are loaded directly 
        y_train (ndarray): the label vector
        data_describtion (str): a describtion of the used data
        model_describtion (str): a describtion of the used model and density model
    '''

    #initiate model
    dice_pred = Dice_predictor(features=feature_extractor.features,add_to_name=model_name,verbose=verbose)

    #Load the features 
    X_train,y_train = feature_extractor.collect_train_data()
    print(len(X_train),len(y_train))

    #train model
    dice_pred.train(X_train,y_train, data_describtion, model_describtion,**kwargs)
    
    if verbose:
        dice_pred.print_description()

def retrain_dice_predictor(model_name='',feature_names=[],dens_model='',dens_add_name='',list_of_paths=[],
                            names_extracted_features=[],y_train=None,data_describtion = '',
                            model_describtion = '',verbose=False,**kwargs):
    '''Retrains a dice predictor model based on features extracted from image-seg pairs
     a standart setting for kwargs is: 
     kwargs = {'solver'='adam','lr'='adaptive','hidden_layer_sizes'=(10,30,50,50,20)}

    Args:
        model_name (str): The name of the model
        feature_names (list(str)):The names of the features to be used
        dens_model (str): The first part of the density model name 
        dens_add_name (str): the second part of the density model name 
            specifying the exact model 
        list_of_paths (list(str)): a list of paths to the directories, where features
            shall be extracted from 
        names_extracted_features (list(str)): names of already extracted features, that are loaded directly 
        y_train (ndarray): the label vector
        data_describtion (str): a describtion of the used data
        model_describtion (str): a describtion of the used model and density model
    '''
    #initiate model
    dice_pred = Dice_predictor(features=feature_names,add_to_name=model_name)

    #get density model
    density = Density_model(model=dens_model,add_to_name=dens_add_name)
    density.load_density()

    #load feature extractor and get features 
    feat_extr = Feature_extractor(density,feature_names)
    features = np.array(feat_extr.get_features_from_paths(list_of_paths))
    loaded_features = feat_extr.load_list_of_feature_vectors(names_extracted_features)
    features = np.append(features,loaded_features)
    X_train = features
    y_train = y_train 

    #train model
    dice_pred.retrain(X_train,y_train, data_describtion, model_describtion,**kwargs)
    
    if verbose:
        dice_pred.print_description()










