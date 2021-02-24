import numpy as np
from mp.utils.Iterators import Component_Iterator, Dataset_Iterator
from skimage.measure import label
from scipy.ndimage import gaussian_filter

def get_array_of_dicescores(seg): 
    '''computes the array of dicescores for the given segmentation,
    it is assumed, that the label of interest is 1 and all other labels are 0.
    More in detail : For every two consecutive slices, the dice score is computed
    throughout the image. In the case of both images being black (no label 1), the dicescore 
    is set to one.

    Args:
        seg (ndarray): the segmentation

    Returns (ndarray): 
    '''
    shape = np.shape(seg)
    nr_slices = shape[0]
    arr_of_dicescores = np.array([])

    first_slide = seg[0, :, :]
    first_ones = np.sum(first_slide)
    for i in range(nr_slices-1):
        second_slide = seg[i+1,:,:]
        second_ones = np.sum(second_slide)
        intersection = np.dot(first_slide.flatten(),
                                 second_slide.flatten())
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
        img (ndarray): the image
        seg (ndarray): its segmentation
        props (dict[str->object]): a regionprops dictionary, c.f. skimage-> regionprops
        
    Returns (list(floats)): a list of two values, the avg dice score and the avg dice score difference'''
    min_row, min_col, min_sl, max_row, max_col, max_sl,  = props.bbox
    cut_seg = seg[min_row:max_row,min_col:max_col,min_sl:max_sl]
    arr_of_dicescores = get_array_of_dicescores(cut_seg)

    # compute the average value of dice score
    dice_avg_value = np.average(arr_of_dicescores)

    # compute the average value of dice score changes between slides
    dice_diff = np.diff(arr_of_dicescores)
    dice_diff_abs = np.absolute(dice_diff)
    dice_diff_avg_value = np.average(dice_diff_abs)

    return [dice_avg_value,dice_diff_avg_value]

def get_int_dens(img, coords):
    '''computes a smoothed density over the intensity values at coords
    Args:
        img (ndarray): the image, whose intensity values we are intrested in
        coords (list(tuples)): a list of the coords in the img, that we want to take the intensities from
            usually the coordinates of a connected component
            
        Returns (ndarray): the density values of the density on the interval [0,1]'''
    intensities = np.array([img[x,y,z] for x,y,z in coords])
    hist= np.histogram(intensities,density=True,bins=np.arange(start=0,stop=1,step=0.001))[0]
    hist = gaussian_filter(hist,sigma=0.005,mode='nearest',truncate=1)
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
    return similarity

def get_similarities(img,seg,props,density_values):
    '''computes the distances of the intensity density of the connected component
    given by props.coords and the given density values

    Args: 
        img (ndarray): an image
        seg (ndarray): its segmentation mask
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

    def __init__(self, density, features=[]):
        self.features = features
        self.nr_features = len(features)
        self.density = density

    def get_features(self,img,seg):
        '''extracts all of self.features for a given image-seg pair
        Args: 
            img (ndarray): an image 
            seg (ndarray): the corresponding mask
            
        Returns: list(numbers): a list of the extracted features'''
        list_features = []
        for feature in self.features:
            feature = self.get_feature(feature,img,seg)
            for attr in feature:
                list_features.append(attr)
        return list_features

    def get_feature(self,feature,img,seg):
        '''Extracts the given feature for the given img-seg pair

        Args: 
            feature (str): The feature to be extracted 
            img (ndarray): the image 
            seg (ndarray): The corresponding mask

        Returns (object): depending on the feature: 
            density_distance -> (float): The average density distance of the connected components and 
                the precomputed density
            dice_scores -> (ndarray): array with two entries, the dice averages and dice_diff averages 
            connected_components -> (int): The number of connected components
        '''
        component_iterator = Component_Iterator(img,seg)
        if feature == 'density_distance':
            density_values = self.density.get_values()
            similarity_scores= component_iterator.iterate(get_similarities,
                    density_values=density_values)
            average = np.mean(np.array(similarity_scores))
            return average
        if feature == 'dice_scores':
            dice_metrices = component_iterator.iterate(get_dice_averages)
            dice_metrices = np.array(dice_metrices)
            dice_metrices = np.mean(dice_metrices,0)
            return dice_metrices
        if feature == 'connected_components':
            _,number_components = label(seg,return_num=True)
            return number_components

    def get_features_from_paths(self,list_paths):
        '''Extracts the features from all img-seg pairs in all paths 

        Args:
            list_paths (list(str)): a list of strings, each string is the path to a dir containing 
                img-seg pairs
        
        Returns: (list(list(numbers))): For every image a list of features in numeric form
        '''
        list_list_features = []
        for path in list_paths:
            if 'UK_Frankfurt2' in path:
                mode = 'UK_Frankfurt2'
            else:
                mode = 'normal'
            ds_iterator = Dataset_Iterator(path,mode=mode)
            output = ds_iterator.iterate_images(self.get_features)
            # output is a list(list(numbers)), list_list_features should be in the same format, so
            # we append every feature-list to the list_list_features single
            for list in output:
                list_list_features.append(list)
        return list_list_features

  
    