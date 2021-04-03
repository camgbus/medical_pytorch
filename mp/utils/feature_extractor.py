import numpy as np
from mp.utils.Iterators import Component_Iterator, Dataset_Iterator
from skimage.measure import label
from scipy.ndimage import gaussian_filter
import os 
import json 
import SimpleITK as sitk
import torch
import torchio

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

def density_similarity(p,q,mode='kl'):
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
        similarity = np.linalg.norm(p-q)
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
    def __init__(self, density, features=[]):
        self.features = features
        self.nr_features = len(features)
        self.density = density
        self.path_to_features = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'extracted_features')
        
        if not os.path.isdir(self.path_to_features):
            os.makedirs(self.path_to_features)
        
        self.density.load_density()

    def get_features(self,img,seg):
        '''extracts all of self.features for a given image-seg pair
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
            density_distance -> (ndarray with one enty): The average density distance of the connected components and 
                the precomputed density
            dice_scores -> (ndarray with two entries): array with two entries, the dice averages and dice_diff averages 
            connected_components -> (ndarray with one int): The number of connected components
        '''
        component_iterator = Component_Iterator(img,seg)
        original_threshhold = component_iterator.threshold
        if feature == 'density_distance':
            density_values = self.density.get_values()
            similarity_scores= component_iterator.iterate(get_similarities,
                    density_values=density_values)
            if not similarity_scores:
                print('Image only has very small components')
                component_iterator.threshold = 0
                similarity_scores= component_iterator.iterate(get_similarities,
                    density_values=density_values)
                component_iterator.threshold = original_threshhold
            if not similarity_scores:
                print('Image has no usable components, no reliable computations can be made')
                similarity_scores = np.array([0])
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
                print('Image has no usable components, no reliable computations can be made')
                dice_metrices = np.array([1,0])
            dice_metrices = np.array(dice_metrices)
            dice_metrices = np.mean(dice_metrices,0)
            return list(dice_metrices)
        if feature == 'connected_components':
            _,number_components = label(seg,return_num=True)
            return number_components

    def get_features_from_paths(self,list_paths,mode='JIP',save=False,save_name=None,save_descr=None):
        '''Extracts the features from all img-seg pairs in all paths 

        Args:
            list_paths (list(str)): a list of strings, each string is the path to a dir containing 
                img-seg pairs in some form
            mode (str) :the saving format of the images
        
        Returns: (2dim ndarray): For every image an array of features
        '''
        list_list_features = []
        for path in list_paths:
            ds_iterator = Dataset_Iterator(path,mode=mode)
            output = ds_iterator.iterate_images(self.get_features)
            # output is a list(list(numbers)), one list for every image; since we want the output list 
            # list_list_features to have the same format, we cant simply append the whole list, but
            # we append every feature-list to the list_list_features over a loop.
            for list in output:
                list_list_features.append(list)
        arr_arr_features = np.array(list_list_features)
        if save:
            self.save_feature_vector(arr_arr_features,save_name,save_descr)
        return arr_arr_features

    def compute_features_id(self,id):
        #get id path depending on global mode 
        if not os.environ["INFERENCE_OR_TRAIN"] == 'train':
            id_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],id)
        else : 
            id_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],id)
        img_path = os.path.join(id_path,'img','img.nii.gz')

        #get path to segmentation mask 
        if os.path.exists(os.path.join(id_path,'seg')):   
            mask_path_short = os.path.join(id_path,'seg')
            self.save_feat_dict_from_paths(id_path,img_path,mask_path_short)
        
        if  os.path.exists(os.path.join(id_path,'pred')):
            for i in range(len(os.listdir(os.path.join(id_path,'pred')))):
                mask_path_short = os.path.join(id_path,'pred','pred_{}'.format(i))
                self.save_feat_dict_from_paths(id_path,img_path,mask_path_short,i)

    def save_feat_dict_from_paths (self,id_path,img_path,mask_path_short,i=0):
        #either we have segmententation or prediction
        if os.path.exists(os.path.join(mask_path_short,'001.nii.gz')):
            mask_path = os.path.join(mask_path_short,'001.nii.gz')
        else : 
            mask_path = os.path.join(id_path,'pred','pred_{}'.format(i),'pred_'+str(i)+'.nii.gz')

        #load image and mask and compute the feature dict
        img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
        mask = torch.tensor(torchio.Image(mask_path, type=torchio.LABEL).numpy())[0]
        feat_dict = {}
        for feat in self.features:
            feat_dict[feat] = self.get_feature(feat,img,mask)
        
        #save the features in a json file 
        feature_save_path = os.path.join(mask_path_short,'features.json')
        print(feat_dict)
        with open (feature_save_path,'w') as f:
            json.dump(feat_dict,f)

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
        



        



  
    