## basic sketch, nothing to see here

import numpy as np
import torchio
from skimage.measure import label, regionprops
import os 
from mp.paths import storage_data_path
import math

def get_array_of_dicescores(img): 

    shape = np.shape(img)
    nr_slices = shape[0]
    arr_of_dicescores = np.array([])

    first_slide = img[0, :, :]
    first_ones = np.sum(first_slide)
    for i in range(nr_slices-1):
        second_slide = img[i+1,:,:]
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

def compute_metrics(seg_path):
    '''gets a 3D segmentation of a Covid consolidation and computes metrices on it.
    Will be used to assess if segmentation is plausible'''

    #load image
    img = torchio.Image(seg_path, type=torchio.LABEL)
    img = img.numpy()
    spacing = img.spacing
    nr_slices = img.shape[2]
    
    # 1. Dice scores
    arr_of_dicescores = get_array_of_dicescores(img)

    # 1.2 compute the average value of dice score
    dice_avg_value = np.average(arr_of_dicescores)

    # 1.3 compute the average value of dice score changes between slides
    dice_diff = np.diff(arr_of_dicescores)
    dice_diff_abs = np.absolute(dice_diff)
    dice_diff_avg_value = np.average(dice_diff_abs)
    


    # 2.Connected components

    # 2.1 get a 3dim array
    shape = img.shape
    img_reshaped = np.resize(img, new_shape=(shape[1], shape[2], shape[3]))

    # 2.2 compute connected components
    labeled_image, nr_components = label(img_reshaped, return_num=True, connectivity=3)

    # 2.3 closer look at single components or avg of 1/2/3/4...
    # area 
    # "anti-convexity" of cc (conv_hull-cc)/area
    # eccentricity
    # euler_number
    # major axis length
    # mean_intensity(how to get intensities in function?)

    
    # 3. Compute volumes 

    # 3.1 Compute volume of segmented tissue in mm3
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    volume = np.sum(img) * voxel_volume

    # 3.2 Compute volume in left and right lung and ratio
    middle = math.floor(nr_slices/2)
    volume_right = np.sum(img[:,:,middle,:]) * voxel_volume
    volume_left = volume - volume_right
    ratio_left_right_lung = volume_left/volume_right


    #4. Compare to more trivial segmentation if possible. Maybe take otsu with pixel value of consolidated area(from centroid intensity of biggest cc) or smth alike 



# Histogram over all values, then decide on interesting values


    


# get the images and into numpy in order to test a bit
global_name = "VESSEL12"
dataset_path = os.path.join(storage_data_path, global_name)
study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name in os.listdir(dataset_path))
x_path = os.path.join(dataset_path, 'VESSEL12_01' + '.nii.gz')
y1_path = os.path.join(dataset_path, 'VESSEL12_01' + '_gt.nii.gz')
y2_path = os.path.join(dataset_path, 'VESSEL12_02' + '_gt.nii.gz')
x = torchio.Image(x_path, type=torchio.INTENSITY)
y1 = torchio.Image(y1_path, type=torchio.LABEL)
y2 = torchio.Image(y2_path, type=torchio.LABEL)






