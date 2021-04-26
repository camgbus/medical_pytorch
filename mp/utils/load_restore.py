# ------------------------------------------------------------------------------
# Functions to save and restore different data types.
# ------------------------------------------------------------------------------

import os
import pickle
import numpy as np
import json
import functools
import torch
import SimpleITK as sitk
import mp.models.cnn.cnn as models

# PICKLE
def pkl_dump(obj, name, path='obj'):
    r"""Saves an object in pickle format."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    pickle.dump(obj, open(path, 'wb'))

def pkl_load(name, path='obj'):
    r"""Restores an object from a pickle file."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    try:
        obj = pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        obj = None
    return obj

# NUMPY
def np_dump(obj, name, path='obj'):
    r"""Saves an object in npy format."""
    if '.npy' not in name:
        name = name + '.npy'
    path = os.path.join(path, name)
    np.save(path, obj)

def np_load(name, path='obj'):
    r"""Restores an object from a npy file."""
    if '.npy' not in name:
        name = name + '.npy'
    path = os.path.join(path, name)
    try:
        obj = np.load(path)
    except FileNotFoundError:
        obj = None
    return obj

# JSON
def save_json(dict_obj, path, name):
    r"""Saves a dictionary in json format."""
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'w') as json_file:
        json.dump(dict_obj, json_file)

def load_json(path, name):
    r"""Restores a dictionary from a json file."""
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'r') as json_file:
        return json.load(json_file)

def save_json_beautiful(data, path, name, sort):
    r"""This function saves a dictionary as a json file at the specified path with indention."""
    if not os.path.exists(path):
        os.makedirs(path)
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'w') as fp:
        json.dump(data, fp, sort_keys=sort, indent=4)

# NIFTY
def nifty_dump(x, name, path):
    r"""Save a tensor of numpy array in nifty format."""
    if 'torch.Tensor' in str(type(x)):
        x = x.detach().cpu().numpy()
    if '.nii' not in name:
        name = name + '.nii.gz'
    # Remove channels dimension and rotate axis so depth first
    if len(x.shape) == 4:
        x = np.moveaxis(x[0], -1, 0)
    assert len(x.shape) == 3
    path = os.path.join(path, name)
    sitk.WriteImage(sitk.GetImageFromArray(x), path)

# OTHERS
def join_path(list):
    r"""From a list of chained directories, forms a path"""
    return functools.reduce(os.path.join, list)

def load_model(model_name, output_features, path, weights):
    r"""This function creates a model and based on path and weights, it loads the
        corresponding state_dict and returns the model."""
    model = getattr(models, model_name)(output_features)    # Same as models.model_name(output_features)
    # If weights, than load the state dict from path
    if weights:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval()
    # Return the model
    return model