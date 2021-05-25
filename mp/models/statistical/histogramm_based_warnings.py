import os 
import json 
import math
import numpy as np 
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from mp.models.densities.density import Density_model
from mp.utils.feature_extractor import Feature_extractor

class histogramm_based_warning():

    def __init__(self) -> None:
        self.path_to_warnings =  os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'hist_based_warnings')

    def load_seg_feature(self,feature):
        features = []
        dens = Density_model()
        feat_extr = Feature_extractor(dens,[feature])
        work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
        for id in os.listdir(work_path):
            id_path = os.path.join(work_path,id)
            seg_path_short = os.path.join(id_path,'seg')
            seg_features_path = os.path.join(seg_path_short,'features.json')
            feat_vec = feat_extr.read_feature_vector(seg_features_path)
            if not np.isnan(np.sum(np.array(feat_vec))):
                features.append(feat_vec)
        return np.array(features)

class hist_based_warning_slice_dice(histogramm_based_warning):

    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(self.path_to_warnings,'slice_dice')
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.path_to_threshholds = os.path.join(self.path,'threshholds.json')
        if os.path.exists(self.path_to_threshholds):
            self.thresh_dict = json.load(self.path_to_threshholds)

    def compute_threshholds(self,percentiles):
        data = self.load_seg_feature('dice_scores')
        thresh_dict = OrderedDict()
        for percent in percentiles:
            threshhold = self.get_threshhold(data,percent)
            thresh_dict[percent]=threshhold
        self.save_threshholds(thresh_dict)

    def get_threshhold(self,data,percent):
        bins = np.arange(np.min(data),1,step=0.001)
        hist, bin_edges = np.histogram(data,bins=bins)
        total_points = np.sum(hist)
        dens = np.array(hist)/total_points
        dens_flipped = np.flip(dens)
        dens_flipped_cumsum = np.cumsum(dens_flipped)
        for i in range(len(dens_flipped_cumsum)):
            if dens_flipped_cumsum[i]>percent:
                return bin_edges[total_points-1-i]

    def save_threshholds(self,thresh_dict):
        with open(self.path_to_threshholds) as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        num_threshholds = len(self.thresh_dict.items()) + 1
        for i,threshhold in enumerate(self.thresh_dict.items()):
            if seg_feature > threshhold:
                return (num_threshholds-i)/num_threshholds
        return 0

class hist_based_warning_conn_comp(histogramm_based_warning):

    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(self.path_to_warnings,'conn_comp')
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.path_to_threshholds = os.path.join(self.path,'threshholds.json')
        if os.path.exists(self.path_to_threshholds):
            self.thresh_dict = json.load(self.path_to_threshholds)

    def compute_threshholds(self,percentiles):
        data = self.load_seg_feature('conn_comp')
        thresh_dict = OrderedDict()
        for percent in percentiles:
            threshhold = self.get_threshhold(data,percent)
            thresh_dict[percent]=threshhold
        self.save_threshholds(thresh_dict)

    def get_threshhold(self,data,percent):
        hist, bin_edges = np.histogram(data,bins=np.arange(0,np.max(data),step=1),density=True)
        cum_hist = np.cumsum(hist)
        for i in range(len(cum_hist)):
            if cum_hist[i]>percent:
                return math.ceil(bin_edges[i])

    def save_threshholds(self,thresh_dict):
        with open(self.path_to_threshholds) as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        num_threshholds = len(self.thresh_dict.items()) + 1
        for i,threshhold in enumerate(self.thresh_dict.items()):
            if seg_feature < threshhold:
                return (num_threshholds-i)/num_threshholds
        return 0

class hist_based_warning_int_mode(histogramm_based_warning):

    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(self.path_to_warnings,'gauss_params')
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.path_to_threshholds = os.path.join(self.path,'threshholds.json')
        if os.path.exists(self.path_to_threshholds):
            self.thresh_dict = json.load(self.path_to_threshholds)

    def compute_threshholds(self,percentiles):
        data = self.load_seg_feature('gauss_params')
        thresh_dict = OrderedDict()
        for percent in percentiles:
            threshholds = self.get_threshhold(data,percent)
            thresh_dict[percent]=threshholds
        self.save_threshholds(thresh_dict)

    def get_threshhold(self,data,percent):

        #first fit a mixture with 2 components to find 2 modes
        gm = GaussianMixture(n_components=2).fit(data)
        if gm.means_[0][0] < gm.means_[1][0]:
            means = [gm.means_[0][0],gm.means_[1][0]]
            vars = [gm.covariances_[0][0][0],gm.covariances_[1][0][0]]
            weights = [gm.weights_[0],gm.weights_[1]]
            # try to balance the steplengths, according to weights and cov
            step_0 = vars[0]*weights[0]
            step_1 = vars[1]*weights[1]
        else:
            means = [gm.means_[1][0],gm.means_[0][0]]
            vars = [gm.covariances_[1][0][0],gm.covariances_[0][0][0]]
            weights = [gm.weights_[1],gm.weights_[0]]
            # try to balance the steplengths, according to weights and std
            step_0 = (vars[0]**(1/2))*(1/20)*weights[0]
            step_1 = (vars[1]**(1/2))*(1/20)*weights[1]
            
        #find the threshholds
        hist_0, bins_0 = np.histogram(data,np.arange(0,1,step_0))
        hist_1, bins_1 = np.histogram(data,np.arange(0,1,step_1))
        number_points = np.sum(hist_0)
        hist_0 = np.array(hist_0)/number_points
        hist_1 = np.array(hist_1)/number_points
        hist = [hist_0,hist_1]
        bins = [bins_0,bins_1]
        mode_0_bin = np.argmax(bins[0]>means[0])
        mode_1_bin = np.argmax(bins[1]>means[1])
        mode_bins = [mode_0_bin,mode_1_bin]

        # if the intervalls are overlapping, inner intervalls are not increased in this case
        overlapping = False
        complete_0 = False
        complete_1 = False 
        i = 0
        mass = 0
        while mass<percent:
            # check whether intervalls are overlapping 
            if bins[1][mode_bins[1]-i] < bins[0][mode_bins[0]+i+1] and not overlapping:
                #add the bigger bin to the mass
                overlapping = True
                if weights[0]>weights[1]:
                    mass = mass + hist[0][mode_bins[0]+i]
                else:
                    mass = mass + hist[1][mode_bins[1]-i]

            if mode_bins[0]-i < 0 or mode_bins[0]+i > len(hist[0]): 
                complete_0 = True 
            if mode_bins[1]-i < 0 or mode_bins[1]+i > len(hist[1]):
                complete_1 = True
            #if both ditributions have reached their end break the loop
            if complete_1 or complete_0 :
                break
            #add masses
            if i == 0:
                mass = hist[0][mode_bins[0]]+hist[1][mode_bins[1]]
            if overlapping:
                mass = mass + hist[0][mode_bins[0]-i]+hist[1][mode_bins[1]+1]
            else:
                mass0 = hist[0][mode_bins[0]-i]+hist[0][mode_bins[0]+i]
                mass1 = hist[1][mode_bins[1]+i]+hist[1][mode_bins[1]-i]
                mass = mass + mass0 + mass1
            i = i + 1
        if overlapping:
            return [bins[0][mode_bins[0]-i+1],bins[1][mode_bins[1]+i]]
        else:
            return [bins[0][mode_bins[0]-i+1],bins[0][mode_bins[0]+i],bins[1][mode_bins[1]-i+1],bins[1][mode_bins[1]+i]]

    def save_threshholds(self,thresh_dict):
        with open(self.path_to_threshholds) as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        num_threshholds = len(self.thresh_dict.items()) +1
        for i,threshholds in enumerate(self.thresh_dict.items()):
            if self.feature_in_threshholds(seg_feature,threshholds):
                return (num_threshholds-i)/num_threshholds
        return 0

    def feature_in_threshholds(self,feature,threshholds):
        if len(threshholds) == 2:
            return ((feature >= threshholds[0]) and (feature <= threshholds[1]))
        if len(threshholds == 4):
            in_first = ((feature >= threshholds[0]) and (feature <= threshholds[1]))
            in_second = ((feature >= threshholds[2]) and (feature <= threshholds[3]))
            return in_first or in_second





    


