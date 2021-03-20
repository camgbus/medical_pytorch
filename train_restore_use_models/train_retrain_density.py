from mp.models.densities.density import Density_model
from mp.utils.intensities import get_intensities, load_intensities
import os
import numpy as np

def train_density(model = '',ending_of_model = '',list_of_paths = [],data_mode='',
                    data_describtion = '', model_describtion = '', 
                    precom_intensities=[], verbose=False, **kwargs):
    '''Trains a density model from a list of given paths to directories, where img-seg pairs can be found
        and stores it

    Args:
        model (str): The name of the model being trained. Gives the first part of the name
        ending_of_model (str): in order to specify model with name 'model' shall be trained ;
            is the second and last part of the density models name
        list_of_paths (list(str)): Every path in this path leads to a directory, that is then later 
            iteratet over in order to gather intensity values
        data_mode (str): the mode in which the data is stored, for further insight, look at Iterators.py
        data_describtion (str): a string to describe the data used for training (e.g. which instances from
            which dataset, when the training was done, etc)
        model_describtion (str): a string to describe the specifications of the density model (e.g. bandwith, 
            other parameters, etc)
        precom_intesities (list(str)): a list of name of already computed intensity values, that are loaded 
            from op_pers_dirs/intensities, dont need .npy ending
        verbose (bool): If consolte outputs should be printed
        **kwargs is used for the specifics of the model to be trained
    '''

    #initialise density model
    density_model = Density_model(model=model,add_to_name=ending_of_model,verbose=verbose)

    #get the intensity values from the images 
    if verbose:
        print('Getting intensity values')
    intensity_values = get_intensities(list_of_paths,mode=data_mode)

    #load already computed intensities and merge the two
    pre_intensities = load_intensities(precom_intensities)
    intensity_values = np.append(intensity_values,pre_intensities)


    #train density model
    if verbose:
        print('Training density model')
    density_model.train_density(intensity_values,
                                data_describtion,model_describtion,**kwargs)

    if verbose:
        density_model.plot_density()

def retrain_density(model = '',ending_of_model = '',list_of_paths = [],
                    add_data_descr = '', add_model_descr = '', 
                    precom_intensities=[], verbose=False, **kwargs):
    '''Trains a density model from a list of given paths to directories, where img-seg pairs can be found
        and stores it, can also use a alread computed intensities

    Args:
        model (str): the type of the model to be retrained 
        ending_of_model (str): which model of the type 'model' shall be retrained 
        list_of_paths (list(str)): Every path in this path leads to a directory, that is then later 
            iteratet over in order to gather intensity values
        data_mode (str): the mode in which the data is stored, for further insight, look at Iterators.py
        data_describtion (str): a string to describe the data used for training (e.g. which instances from
            which dataset, when the training was done, etc)
        model_describtion (str): a string to describe the specifications of the density model (e.g. bandwith, 
            other parameters, etc)
        precom_intesities (list(str)): a list of name of already computed intensity values, that are loaded 
            from op_pers_dirs/intensities, dont need .npy ending
        verbose (bool): If console outputs should be printed
        **kwargs is used for the specifics of the model to be trained
    '''

    #initialise density model
    density_model = Density_model(model=model,add_to_name=ending_of_model)

    #get the intensity values from the images 
    if verbose:
        print('Getting intensity values')
    intensity_values = get_intensities(list_of_paths)

    #load already computed intensities and merge the two
    pre_intensities = load_intensities(precom_intensities)
    intensity_values = np.append(intensity_values,pre_intensities)

    #train density model
    if verbose:
        print('Retraining density model')
    density_model.retrain_density(intensity_values,add_data_descr,
                                    add_model_descr,**kwargs)

    if verbose:
        density_model.plot_density()
