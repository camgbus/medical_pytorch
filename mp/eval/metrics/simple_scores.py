import numpy as np

def dice_score(target,prediction):
    '''Computes the dice score between two binary segmentation masks
    Args:
        target (nd.array): the ground truth
        prediction (nd.array): the prediction
        
    Returns (float): the dice score'''
    intersection = np.dot(target.flatten(),prediction.flatten())
    target_ones, prediction_ones = np.sum(target), np.sum(prediction)
    if target_ones == 0 and prediction_ones == 0 : 
        dice_score = 0
    else:
        dice_score = 2*intersection/(target_ones+prediction_ones)
    return dice_score

