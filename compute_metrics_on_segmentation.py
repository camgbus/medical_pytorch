## basic sketch, nothing to see here

import numpy as np
import torchio
from skimage.measure import label, regionprops
import os 
from mp.paths import storage_data_path
import math
import pickle
from scipy.ndimage import gaussian_filter
from Iterators import Dataset_Iterator,Component_Iterator

def get_array_of_dicescores(seg): 

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
    intensities = np.array([img[x,y,z] for x,y,z in coords])
    hist = np.histogram(intensities,density=True,bins=np.arange(start=-1024,stop=3072))[0]
    hist = gaussian_filter(hist,sigma=20,mode='nearest',truncate=1)
    return hist

def density_similarity(p,q,mode='kl'):
    similarity = 0
    if mode == 'kl':
        for i in range(len(p)):
            pi = p[i]
            qi = q[i]
            if (pi < 0.000001 ):
                continue # equal as setting the summand to zero
            elif(qi == 0):
                continue
            else :
                summand = pi * (np.log(pi/qi))
                similarity += summand
    return similarity

def get_similarities(img,seg,props,density_values):
    coords = props.coords
    comp_intenity_density = get_int_dens(img,coords)
    similarity = density_similarity(density_values,comp_intenity_density)
    return similarity
    
def compute_metrics(img_path,seg_path):
    '''gets a 3D segmentation of a Covid consolidation and computes metrices on it.
    Will be used to assess if segmentation is plausible'''

    #load image and segmentation
    img = torchio.Image(img_path, type=torchio.INTENSITY)
    seg = torchio.Image(seg_path, type=torchio.LABEL)
    img = img.numpy()[0]
    seg = seg.numpy()[0]
    shape = np.shape(img)
    nr_slices = shape[2]

    # 1.Compare histograms of the components to intensity densities of segmentations
    density = pickle.load(open(os.path.join('storage','statistics','UK_Frankfurt2','density_estimation','kde_gauss_bw_20.sav'),'rb'))
    density_values = np.exp(density.score_samples(np.reshape(np.arange(start=-1024,stop=3071),(-1,1)))) #for computation of density similarity
    component_iterator = Component_Iterator(img,seg)
    similarity_scores_densities = component_iterator.iterate(get_similarities,density_values=density_values)
    average = np.mean(np.arry(similarity_scores_densities))

    # 2.Dice scores for the components
     dice_metrices = component_iterator.iterate(get_dice_averages)

    # 3.Number of connected components 
     _,number_components = label(seg)


    return average , similarity_scores_densities , dice_metrices , number_components




# #small testing 
# path = os.path.join('downloads','UK_Frankfurt2','KGU-1D1840AEB676')
# img_path = os.path.join(path,'image.nii.gz')
# seg_path = os.path.join(path,'mask.nii.gz')
# print(compute_metrics(img_path,seg_path))







