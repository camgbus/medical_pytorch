import numpy as np
from mp.utils.Iterators import Component_Iterator, Dataset_Iterator
from skimage.measure import label
from scipy.ndimage import gaussian_filter
import os 
import json 
import SimpleITK as sitk
import torch
import torchio
from mp.models.densities.density import Density_model
import datetime

def get_array_of_dicescores(seg): 
    '''computes the array of dicescores for the given segmentation,
    it is assumed, that the label of interest is 1 and all other labels are 0.
    More in detail : For every two consecutive slices, the dice score is computed
    throughout the image. In the case of both images being black (no label 1), the dicescore 
    is set to one.

    Args:
        seg (torch.Tensor): the segmentation

    Returns (ndarray): array of dicescores
    '''
    shape = np.shape(seg)
    nr_slices = shape[0]
    arr_of_dicescores = np.array([])
    
    first_slide = seg[0, :, :]
    first_ones = torch.sum(first_slide)
    for i in range(nr_slices-1):
        second_slide = seg[i+1,:,:]
        second_ones = torch.sum(second_slide)
        intersection = torch.dot(torch.flatten(first_slide),
                                torch.flatten(second_slide))
        # if two consecutive slices are black, set dice score to one
        # leads to ignoring the score
        if not(first_ones+second_ones == 0):
            dice_score = 2*intersection / (first_ones+second_ones)
        else:
            dice_score = 1
        # if two consecutive slides are identical (having dicescore = 1), it is assumed, that they are copied
        # or completly black and thus dont count towards the overall Dicescore
        if not dice_score == 1:
            arr_of_dicescores = np.append(
                arr_of_dicescores, np.array([dice_score]))
        # update index
        first_slide = second_slide
        first_ones = second_ones
    return arr_of_dicescores

def get_dice_averages(img,seg,props):
    '''Computes the average dice score for a connected component of the 
    given img-seg pair. Also computes the average differences between the dice scores 
    and computes that average, because it was observed, that in bad artificial bad segmentations,
    these dice scores had a more rough graph, then good segmentations, thus it is used as feature.
    
    Args: 
        img (torch.Tensor): the image
        seg (torch.Tensor): its segmentation
        props (dict[str->object]): a regionprops dictionary, c.f. skimage-> regionprops
        
    Returns (list(floats)): a list of two values, the avg dice score and the avg dice score difference'''
    min_row, min_col, min_sl, max_row, max_col, max_sl,  = props.bbox
    cut_seg = seg[min_row:max_row,min_col:max_col,min_sl:max_sl]
    arr_of_dicescores = get_array_of_dicescores(cut_seg)

    # compute the average value of dice score
    dice_avg_value = np.average(arr_of_dicescores)

    # compute the average value of dice score changes between slides
    # check for connected component, that is only on one slice
    if len(arr_of_dicescores) < 10:
        dice_diff_avg_value = 1 
    else:
        dice_diff = np.diff(arr_of_dicescores)
        dice_diff_abs = np.absolute(dice_diff)
        dice_diff_avg_value = np.average(dice_diff_abs)

    return [dice_avg_value,dice_diff_avg_value]

def get_int_dens(img, coords):
    '''computes a smoothed density over the intensity values at coords
    Args:
        img (torch.Tensor): the image, whose intensity values we are intrested in
        coords (list(tuples)): a list of the coords in the img, that we want to take the intensities from
            usually the coordinates of a connected component
            
        Returns (ndarray): the density values of the density on the interval [0,1]'''
    rng = np.random.default_rng()
    if len(coords) > 5000:
        coords = rng.choice(coords,5000,replace=False,axis=0)
    intensities = np.array([img[x,y,z] for x,y,z in coords])
    hist= np.histogram(intensities,density=True,bins=np.arange(start=0,stop=1.001,step=0.001))[0]
    hist = gaussian_filter(hist,sigma=10,mode='nearest',truncate=4)
    return hist

def density_similarity(p,q,mode='l2'):
    '''Computes the distance of two densities p,q, that are given through values 
    
    Args:
        p and q (ndarray): arrays of equal length, containing the values of the densities 
            on the interval [0,1]
        model (str): which mode to use for computation of distance
    
    Returns (float): a distance between the two densities
    '''
    similarity = 0
    assert(len(p)==len(q))
    if mode == 'kl':
        for i in range(len(p)):
            pi = p[i]
            qi = q[i]
            if (pi < 0.0000001 ):
                continue # equal as setting the summand to zero
            elif(qi < 0.000000001):
                qi = 0.000000001
                summand = pi * (np.log(pi/qi))
                similarity += summand
            else :
                summand = pi * (np.log(pi/qi))
                similarity += summand
    if mode == 'l2':
        diff = np.absolute(p-q)
        for i in range(len(diff)):
            if diff[i]<4:
                diff[i]=0
        similarity = np.sum(diff**2)
    return similarity

def get_similarities(img,seg,props,density_values):
    '''computes the distances of the intensity density of the connected component
    given by props.coords and the given density values

    Args: 
        img (torch.Tensor): an image
        seg (torch.Tensor): its segmentation mask, unused but needed for  
            compability
        props (dict[str->object]): the regoinprop dictionary, for further 
            information see skimage->regionprops
        density_values (ndarray): Array containing the density values of a learned 
            density, which we want to compare to. should be values for the interval [0,1]
        
    Return (float): the distance between the two densities, computed by KL-divergence
    '''
    coords = props.coords
    comp_intenity_density = get_int_dens(img,coords)
    similarity = density_similarity(density_values,comp_intenity_density)
    return similarity

class Feature_extractor():
    '''A class for extracting feature of img-seg pairs and get arrays of features

    Args: 
        density (Density_model): a density model, with a loaded density
        feature (list(str)): Every string in the list is for one feature: 
            -density_distance : computes the distance between the densities of the img-seg
                pair and the precomputed density 
            -dice_scores : computes the avg dice scores of the components and the avg 
                difference of dice scores 
            -connected components : the number of connected components in the seg
    '''
    def __init__(self, density, features=['density_distance','dice_scores','connected_components']):
        self.features = features
        self.nr_features = len(features)

        self.density = density
        self.density.load_density_values()
        self.density_values = self.density.density_values
        self.path_to_features = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'extracted_features') #deprecated
        
        if not os.path.isdir(self.path_to_features):
            os.makedirs(self.path_to_features)
        
    def get_features(self,img,seg):
        '''
        !!!! DEPRICATED !!!!!extracts all of self.features for a given image-seg pair
        assumes, that each extracteted feature is an integer or a list/array of integers
        Args: 
            img (ndarray): an image 
            seg (ndarray): the corresponding mask
            
        Returns: ndarray(numbers): arry of the extracted features'''
        list_features = []
        for feature in self.features:
            feature = self.get_feature(feature,img,seg)
            for attr in feature:
                list_features.append(attr)
        arr_features = np.array(list_features)
        arr_features = np.around(arr_features,decimals=4)
        return arr_features

    def get_feature(self,feature,img,seg):
        '''Extracts the given feature for the given img-seg pair

        Args: 
            feature (str): The feature to be extracted 
            img (ndarray): the image 
            seg (ndarray): The corresponding mask

        Returns (object): depending on the feature: 
            density_distance -> (integer): The average density distance of the connected components and 
                the precomputed density
            dice_scores -> (ndarray with two entries): array with two entries, the dice averages and dice_diff averages 
            connected_components -> (integer): The number of connected components
        '''
        component_iterator = Component_Iterator(img,seg)
        original_threshhold = component_iterator.threshold
        if feature == 'density_distance':
            density_values = self.density_values
            similarity_scores= component_iterator.iterate(get_similarities,
                    density_values=density_values)
            if not similarity_scores:
                print('Image only has very small components')
                component_iterator.threshold = 0
                similarity_scores= component_iterator.iterate(get_similarities,
                    density_values=density_values)
                component_iterator.threshold = original_threshhold
            if not similarity_scores:
                print('Image has no usable components, no reliable computations can be made for density_dis')
                similarity_scores = 0
            average = np.mean(np.array(similarity_scores))
            return average
        if feature == 'dice_scores':
            dice_metrices = component_iterator.iterate(get_dice_averages)
            if not dice_metrices:
                print('Image only has very small components')
                component_iterator.threshold = 0
                dice_metrices = component_iterator.iterate(get_dice_averages)
                component_iterator.threshold = original_threshhold
            if not dice_metrices:
                print('Image has no usable components, no reliable computations can be made for dice')
                return [1,0]
            dice_metrices = np.array(dice_metrices)
            dice_metrices = np.mean(dice_metrices,0)
            return list(dice_metrices)
        if feature == 'connected_components':
            _,number_components = label(seg,return_num=True)
            return number_components
        
    def compute_features_id(self,id):
        '''Computes all features for the img-seg and img-pred pairs (if existing)
        and saves them in the preprocessed_dir/.../id/...
        Args:
            id (str): the id of the patient to compute the features for
        '''
        #get id path depending on global mode 
        if not os.environ["INFERENCE_OR_TRAIN"] == 'train':
            id_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],id)
        else : 
            id_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],id)
        img_path = os.path.join(id_path,'img','img.nii.gz')
        
        #get the features for the predictions
        all_pred_path = os.path.join(id_path,'pred')
        if  os.path.exists(all_pred_path):
            for model in os.listdir(all_pred_path):
                mask_path_short = os.path.join(id_path,'pred',model)
                self.save_feat_dict_from_paths(img_path,mask_path_short)

    def save_feat_dict_from_paths (self,img_path,mask_path_short):
        '''takes the paths to an img and a mask and a number for a prediction, 
        computes a dictionary of features and saves them in a dictionary 
        in the path of the prediction
        Args: 
            id_path(str): The path to the target folder of the id 
            img_path(str):the path to the img
            mask_path_short(str): The string to either the dir of the seg
                                        gets used, depending if we use seg or 
                                        pred
            i (int): the number of the prediction whose mask we want to use
                                        only gets use if we use prediction 
        '''
        mask_path = os.path.join(mask_path_short,'pred.nii.gz')

        #load image and mask and compute the feature dict
        img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
        mask = torch.tensor(torchio.Image(mask_path, type=torchio.LABEL).numpy())[0]
        feat_dict = {}
        for feat in self.features:
            feat_dict[feat] = self.get_feature(feat,img,mask)
        
        #save the features in a json file 
        feature_save_path = os.path.join(mask_path_short,'features.json')
        with open (feature_save_path,'w') as f:
            json.dump(feat_dict,f)

    def collect_train_data(self,save_as_vector=True):
        if os.environ["INFERENCE_OR_TRAIN"] == 'train':
            all_features = []
            labels = []
            path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"] ,os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
            for id in os.listdir(path):
                all_pred_path = os.path.join(path,id,'pred')
                if os.path.exists(all_pred_path):
                    for model in os.listdir(all_pred_path):
                        pred_path = os.path.join(all_pred_path,model)
                        feature_path = os.path.join(pred_path,'features.json')
                        label_path = os.path.join(pred_path,'dice_score.json')
                        feature_vec = self.read_feature_vector(feature_path)
                        label = self.read_prediction_label(label_path)
                        #filter out any nans 
                        if np.isnan(np.sum(np.array(feature_vec))):
                            pass 
                        else:
                            all_features.append(feature_vec)
                            labels.append(label)
            if save_as_vector:
                name = 'extracted_features_last_training'
                descr = 'extracted features on {}'.format(datetime.date.today())
                self.save_feature_vector(all_features,name,descr)
        else:
            print('This method is only for train time')
            RuntimeError
        return np.array(all_features),np.array(labels)

    def collect_train_data_split(self,save_as_vector=True):
        original_os_path = os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"]
        os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = os.path.join(original_os_path,'train')
        X_train,y_train =self.collect_train_data(save_as_vector)
        os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = os.path.join(original_os_path,'test')
        X_test,y_test =self.collect_train_data(save_as_vector)
        os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = original_os_path
        return X_train,X_test,y_train,y_test

    def read_feature_vector(self,feature_path):
        feature_vec = []
        with open(feature_path,'r') as file:
            feat_dict = json.load(file)
            for feature_name in self.features:
                feature = feat_dict[feature_name]
                if isinstance(feature,list): 
                    for entry in feature:
                        feature_vec.append(entry)
                else:
                    feature_vec.append(feature)
        return feature_vec

    def read_prediction_label(self,label_path):
        with open(label_path,'r') as file:
            label = json.load(file)
        return label

    def load_feature_vector(self,name):
        path_to_features_save = os.path.join(self.path_to_features,name+'.npy')
        feature_vector = np.load(path_to_features_save)
        return feature_vector

    def load_list_of_feature_vectors(self,flist):
        length = len(flist)
        name = flist[0]
        features = self.load_feature_vector(name)
        for i in range(1,length):
            name = flist[i]
            feature_vec = self.load_feature_vector(name)
            features = np.append(features,feature_vec)
        return features

    def save_feature_vector(self,feature_vector,name,describtion):

        if name == None or describtion == None:
            print('features wont get saved, due to missing name and or describtion. Hold your data clean')
        else:
            #set places to save
            path_to_features_save = os.path.join(self.path_to_features,name+'.npy')
            path_to_features_descr = os.path.join(self.path_to_features,name+'_descr.txt')

            # save features
            np.save(path_to_features_save,feature_vector)

            #save describtion
            with open(path_to_features_descr,'w') as file:
                file.write(describtion)
                file.write("\n")
                file.write("With features: {}".format(self.features))
    
    # def bring_features_into_intervall(self,features_vec):
    #     scores = []
    #     precom_features = self.load_feature_vector('extracted_features_last_training')
    #     for i,feature in enumerate(features_vec):
    #         #sort feature vector
    #         #find place 
    #         #return place
            



        



  
    