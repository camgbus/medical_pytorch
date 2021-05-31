import os 
import json 
import math
import numpy as np 
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from seg_qual_model_eval import filter_feature_extr, extract_features_train_id_od
from mp.models.densities.density import Density_model # pylint: disable=import-error
from mp.utils.feature_extractor import Feature_extractor # pylint: disable=import-error
import matplotlib.pyplot as plt

def filter_seg_feature_extr(id):
    '''function to decide which split a prediction should be put in.
    Is infered from the id(giving the dataset of the image) and model
    (the dataset the predicting model was trained on)
    Args:
        id(str):giving the dataset of the image
        model(str):the dataset the predicting model was trained on

    Returns(str):the split for the pair of id and model, can be other, which can be 
        category of unused data'''
    # train data 
    if id[:7] in ['Task740','Task541']:
        return 'train'
    # id data
    if id[:7] in ['Task741']:
        return 'gc'
    if id[:7] in ['Task542']:
        return 'frank'
    # od data 
    if id[:7] in ['Task200','Task201']:
        return 'mosmed'
    if id[:7] in ['Task100','Task101']:
        return 'radio'
    
    return 'other'

def extract_seg_features_train_id_od(filter,splits,used_feat=[0,1,2]):
    '''extracts the features from the preprocess dictionary, and returns a  list,
    in which each entry corresponds to the features/labels of one split
    Args:
        filter(function: (str,str)->str): should be modified filter_feature_extr
        splits(list(str)):the name of the splits
        used_feat(list(int)):the used features, see in main() for more depth

    Returns(list): list, in which each entry corresponds to the features/labels of one split'''
    X = []
    for i in range(len(splits)):
        X.append([])
    feat_extr = Feature_extractor()
    work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    for id in os.listdir(work_path):
        split = filter(id)
        for i,s in enumerate(splits):
            if split == s:
                feature_path = os.path.join(work_path,id,'seg','features.json')
                feat_vec = feat_extr.read_feature_vector(feature_path)
                feat_vec = [feat_vec[index] for index in used_feat]
                if np.isnan(np.sum(np.array(feat_vec))) or feat_vec[0]>100000:
                    pass 
                else:
                    X[i].append(feat_vec)
                            
    return X

class histogramm_based_warning():

    def __init__(self) -> None:
        self.path_to_warnings =  os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'hist_based_warnings')

    def load_seg_feature(self,feature):
        if feature == 'dice_scores':
            feat = 0
        if feature == 'conn_comp':
            feat = 1
        if feature == 'gauss_params':
            feat = 2
        splits = ['train','gc','frank','mosmed','radio']
        X = extract_seg_features_train_id_od(filter_seg_feature_extr,splits,used_feat=[0,1,2])
        X = np.array(X[0])
        return X[:,feat]

class hist_based_warning_slice_dice(histogramm_based_warning):

    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(self.path_to_warnings,'slice_dice')
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.path_to_threshholds = os.path.join(self.path,'threshholds.json')
        if os.path.exists(self.path_to_threshholds):
            with open(self.path_to_threshholds,'r') as file:
                self.thresh_dict = json.load(file)

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
        len_hist = len(hist)
        for i in range(len(hist)):
            weight = np.sum(dens[len_hist-i:len_hist])
            if weight >= percent:
                return bin_edges[len_hist-i]

    def save_threshholds(self,thresh_dict):
        with open(self.path_to_threshholds,'w') as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        num_threshholds = len(self.thresh_dict.values()) + 1
        for i,(_,threshhold) in enumerate(self.thresh_dict.items()):
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
            with open(self.path_to_threshholds,'r') as file:
                self.thresh_dict = json.load(file)

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
        with open(self.path_to_threshholds,'w') as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        num_threshholds = len(self.thresh_dict.items()) + 1
        for i,(_,threshhold) in enumerate(self.thresh_dict.items()):
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
            with open(self.path_to_threshholds,'r') as file:
                self.thresh_dict = json.load(file)

    def compute_threshholds(self,percentiles):
        data = self.load_seg_feature('gauss_params')
        thresh_dict = OrderedDict()
        for percent in percentiles:
            threshholds = self.get_threshhold(data,percent)
            thresh_dict[percent]=threshholds
        self.save_threshholds(thresh_dict)

    def get_threshhold(self,data,percent):
        #first fit a mixture with 2 components to find 2 modes
        data = np.reshape(data,(-1,1))
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
        with open(self.path_to_threshholds,'w') as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        num_threshholds = len(self.thresh_dict.items()) +1
        for i,(_,threshholds) in enumerate(self.thresh_dict.items()):
            if self.feature_in_threshholds(seg_feature,threshholds):
                return (num_threshholds-i)/num_threshholds
        return 0

    def feature_in_threshholds(self,feature,threshholds):
        if len(threshholds) == 2:
            return ((feature >= threshholds[0]) and (feature <= threshholds[1]))
        if len(threshholds) == 4:
            in_first = ((feature >= threshholds[0]) and (feature <= threshholds[1]))
            in_second = ((feature >= threshholds[2]) and (feature <= threshholds[3]))
            return in_first or in_second



def get_cum_metric(feat_list,warn_s,warn_c,warn_m):
    labels = np.array([warn_s.label_seg(feat_list[0]),warn_c.label_seg(feat_list[1]),warn_m.label_seg(feat_list[2])])
    return np.sum(labels)

def plot_hist_metric_per_split(X,y,splits,warn_s,warn_c,warn_m):

    for j,split in enumerate(splits):
        X[j] = [get_cum_metric(X[j][k],warn_s,warn_c,warn_m) for k in range(len(X[j]))]

    for j,split in enumerate(splits):
        plt.scatter(X[j],y[j],label=split)
    plt.xlabel('cumulated metric')
    plt.ylabel('dice score of prediction')
    plt.legend(loc='upper left')
    plt.show()



def main(used_feat=[0,1,2]):
    seg_splits = ['train','gc','frank','mosmed','radio']
    X_seg = extract_seg_features_train_id_od(filter_seg_feature_extr,seg_splits,used_feat)

    splits = ['train','gc_gc','gc_frank','gc_mosmed','gc_radio']
    X,y = extract_features_train_id_od(filter_feature_extr,splits,used_feat)

    percentiles = [0.8,0.9,0.95,0.99]
    warn_slice = hist_based_warning_slice_dice()
    warn_slice.compute_threshholds(percentiles)
    warn_conn = hist_based_warning_conn_comp()
    warn_conn.compute_threshholds(percentiles)
    warn_mode = hist_based_warning_int_mode()
    warn_mode.compute_threshholds(percentiles)

    plot_hist_metric_per_split(X,y,splits,warn_slice,warn_conn,warn_mode)


if __name__ == '__main__':
    main(used_feat=[0,1,2])


    


