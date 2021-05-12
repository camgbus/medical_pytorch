import os
from mp.paths import JIP_dir
import numpy as np
from skimage.measure import label,regionprops
import matplotlib.pyplot as plt
import math 
#set environmental variables
#for data_dirs folder, nothing changed compared to Simons version 
os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
os.environ["OPERATOR_IN_DIR"] = "input"
os.environ["OPERATOR_OUT_DIR"] = "output"
os.environ["OPERATOR_TEMP_DIR"] = "temp"
os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')

# preprocessing dir and subfolders 
os.environ["PREPROCESSED_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'preprocessed_dirs')
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"] = "output_scaled"
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = "output_scaled_train"

#dir where train data for intensites is stored (this only needs to be trains_dirs, but since i have more 
# datasets, another subfolder is here)
os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')

#ignore
##below is for christian only, used for older data structures where models are trained on
os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('Covid-RACOON','All images and labels')
os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('Covid-RACOON','All predictions')
#os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('gt_small')
#os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('pred_small')

#which mode is active either 'train' or 'inference' 
os.environ["INFERENCE_OR_TRAIN"] = 'train'

#ignore
# the ending of the image files in train_dir is only for older datasets
os.environ["INPUT_FILE_ENDING"] = 'nii.gz'

from mp.utils.preprocess_utility_functions import basic_preprocessing
from mp.utils.preprocess_utility_functions import extract_features_all_data,compute_all_prediction_dice_scores
from train_restore_use_models.train_int_based_quantifier import train_dice_predictor
from mp.utils.intensities import sample_intensities
from mp.models.densities.density import Density_model
from mp.utils.feature_extractor import Feature_extractor

## work on variable 3, connected components
def poisson(lam,b):
    return ((lam**b) * np.exp(-lam)) /np.math.factorial(b)

def gaussian(mu,std,b):
    return ((1 / (np.sqrt(2 * np.pi) * std)) * 
                np.exp(-0.5 * (1 / std * (b - mu))**2))
def load_seg_features():
    features = []
    dens = Density_model()
    feat_extr = Feature_extractor(dens)
    work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    for id in os.listdir(work_path):
        id_path = os.path.join(work_path,id)
        seg_path_short = os.path.join(id_path,'seg')
        seg_features_path = os.path.join(seg_path_short,'features.json')
        feat_vec = feat_extr.read_feature_vector(seg_features_path)
        if not np.isnan(np.sum(np.array(feat_vec))):
            features.append(feat_vec)
    return np.array(features)

def plot_conn_comp(data,save=True,fit=True):
    save_path = os.path.join('storage','Results','histogramms','connected components poisson')
    mu = np.mean(data)
    std = np.std(data)
    _ , bins, _ = plt.hist(data,75,(0,np.max(data)),density=True)

    if fit:
        y = [poisson(mu,b) for b in bins]
        plt.plot(bins,y,'--')

    plt.title('Segmentations connected components gaussian')
    if save:
        plt.savefig(save_path)
    plt.show()

def conn_comp_n_percent(data,percent):
    hist, bin_edges = np.histogram(data,bins=np.arange(0,np.max(data),step=1),density=True)
    cum_hist = np.cumsum(hist)
    for i in range(len(cum_hist)):
        if cum_hist[i]>percent:
            return math.ceil(bin_edges[i])

## work for avg slice dice
def plot_slice_dice_hist(data,save=False):
    _ , _, _ = plt.hist(data,50,(np.min(data),1),density=True)
    plt.show()

def slice_dice_n_percent(data,percent):
    bins = np.arange(np.min(data),1,step=0.001)
    hist, bin_edges = np.histogram(data,bins=bins)
    total_points = np.sum(hist)
    dens = np.array(hist)/total_points
    dens_flipped = np.flip(dens)
    dens_flipped_cumsum = np.cumsum(dens_flipped)
    for i in range(len(dens_flipped_cumsum)):
        if dens_flipped_cumsum[i]>percent:
            return bin_edges[total_points-1-i]
            

def main(conn_comp=True,slice_dice=True):
    features = load_seg_features()
    if conn_comp:
        ## variable 3, connected comp
        plot_conn_comp(features[:,3],False,False)
        thresh = conn_comp_n_percent(features[:,3],0.99)
        print('The recommended threshold for connected components is {}'.format(thresh))
    if slice_dice:
        data = features[:,1]
        plot_slice_dice_hist(data)
        thresh = slice_dice_n_percent(data,0.99)
        print('The recommended threshold for slice dices is {}'.format(thresh))



if __name__ == "__main__":
    main(True,True)