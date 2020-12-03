## 1. Imports 
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_mr_prostate_decathlon import DecathlonProstateT2
from mp.data.datasets.ds_mr_lung_VESSEL12 import VESSEL12
import mp.visualization.visualize_imgs as vis
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump

import os 
import SimpleITK as sitk
import numpy as np
import torchio
import math

from mp.eval.inference.predictor import Predictor2D
from mp.eval.inference.predictor import Predictor3D

import mp.data.pytorch.transformation as trans

import matplotlib.pyplot as plt 

## 2.Hyperparams of experiment run
USE_SERVER = False 
PATH_TO_NEW_SEGMENTATION = os.path.join('storage','data','VESSEL12_Segmentation_with_dec_Prostate_model_epoch_5')
PLOT_RESULTS = True
TOOL_FOR_DIM_RECTION = 'TrunctatedSVD' # should be 'TrunctatedSVD' or 'PCA'
CUDA_DEVICE = 0

## 3.Load segmentation agent from state
if USE_SERVER:
    device = 'cuda:'+str(CUDA_DEVICE) # device to use for computation config['device']
else: 
    device = torch.device('cpu') 

# fill in information for segmentation model
input_shape = (1, 256, 256)  # input shape of segmenting model config['input_shape']
label_names = ['lung','background']
nr_labels = 2 # number of labels in data

# load model
model = UNet2D(input_shape, nr_labels)
model.to(device)
agent = SegmentationAgent(model=model, label_names=label_names, device=device)
agent.restore_state(os.path.join('storage','exp','test_exp','0','states'),'epoch_5')


# ## get data to test some things 
# data = Data()
# data.add_dataset(DecathlonProstateT2(merge_labels=True))

# ## load predictor 
# subject_ix = 3
# predictor = Predictor2D(data.datasets['DecathlonProstateT2'].instances,size=(1,256,256,18))
# pred = predictor.get_subject_prediction(agent, subject_ix)
# pred = pred.permute(0,3,1,2)
# pred = pred.numpy()
# shape = pred.shape
# pred = np.resize(pred,(shape[1],shape[2],shape[3]))
# sitk.WriteImage(sitk.GetImageFromArray(pred), os.path.join('ignored','testing_' + str(subject_ix) + '_gt.nii.gz'))


## 4. get the lung images and segment them using the agent or load them if already there
if not os.path.isdir(PATH_TO_NEW_SEGMENTATION):
    os.makedirs(PATH_TO_NEW_SEGMENTATION)
    print('Segmenting Images')
    global_name = "VESSEL12"
    dataset_path = os.path.join('storage','data', global_name)
    for id,subject_ix in enumerate(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']):
        x_path = os.path.join(dataset_path, 'VESSEL12_'+ subject_ix + '.nii.gz')
        x = torch.tensor(torchio.Image(x_path, type=torchio.INTENSITY).numpy())
        original_size_2d = x.shape[:3] 
        original_size = x.shape
        x = x.permute(3,0,1,2)
        pred = []
        with torch.no_grad():
            for slice_idx in range(len(x)):
                inputs = trans.resize_2d(x[slice_idx], size=(1,256,256)).to(agent.device)
                inputs = torch.unsqueeze(inputs, 0)
                slice_pred = agent.predict(inputs).float()
                pred.append(trans.resize_2d(slice_pred, size=original_size_2d, label=True))
        # Merge slices and rotate so depth last
        pred = torch.stack(pred, dim=0) # depth,channel,weight,height
        pred = pred.permute(1, 2, 3, 0) # channel,weight,height,depth
        assert original_size == pred.shape
        pred = pred.numpy()
        shape = pred.shape
        pred = np.resize(pred,(shape[1],shape[2],shape[3]))
        sitk.WriteImage(sitk.GetImageFromArray(pred), os.path.join(PATH_TO_NEW_SEGMENTATION,'segmented_lung_' + subject_ix + '_gt.nii.gz'))
    print('Images segmented and saved')
else:
    print('Images can be loaded directly')


## 5.Compute Dice Scores 

# plotting
if PLOT_RESULTS:
    fig, axs = plt.subplots(4,5, sharey=True)
    fig.suptitle('Dice scores')

list_of_all_dicescores =[]
segmented_path = PATH_TO_NEW_SEGMENTATION
for id,j in enumerate(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']):
    # load images and initialise array to save results
    path=os.path.join(segmented_path, 'segmented_lung_'+ j + '_gt.nii.gz')
    img = torchio.Image(path, type=torchio.LABEL)
    img = img.numpy()
    nr_slices = img.shape[2]
    arr_of_dicescores = np.array([])


    # compute dice scores
    first_slide = img[:,:,0,:]
    first_ones = np.sum(first_slide)
    for i in range(nr_slices-1):
        second_slide = img[:,:,i+1,:]
        second_ones = np.sum(second_slide)
        intersection = np.dot(first_slide.flatten(),second_slide.flatten())
        dice_score = 2*intersection / ( first_ones+second_ones)
        #if two consecutive slides are identical (having dicescore = 1), it is assumed, that they are copied 
        #or completly black and thus dont count towards the overall Dicescore
        if not dice_score == 1:
            arr_of_dicescores = np.append(arr_of_dicescores,np.array([dice_score]))
        #update index
        first_slide = second_slide
        first_ones = second_ones
    if PLOT_RESULTS:
        axs[math.floor(id/5), id % 5].plot(arr_of_dicescores)
    list_of_all_dicescores.append(arr_of_dicescores)
if PLOT_RESULTS:
    plt.show()


## 6. Perform Trunctated SVD or PCA for dimensionality reduction on segmented data

# imports 
import sklearn
from sklearn.decomposition import TruncatedSVD,PCA

# set size for rescaling
new_size = 125 #new height and width
new_slices = 125 # new number of slices

# load data on which decomposition is trained 
data = Data()
data.add_dataset(VESSEL12())

## prepare partitioning into parts
nr_instances = 0
for dataset in data.datasets.values():
    nr_instances = nr_instances + dataset.size

nr_parts = 5
slices_per_part = math.floor(nr_instances*new_slices*(1/nr_parts))
slices_per_image_part = math.floor(new_slices/nr_parts) ##maybe change partitioning later 
assert(isinstance(slices_per_image_part,int))
assert(isinstance(slices_per_part,int))

# get all segmentation instances into an array, in order to use sklearn trunctated SVD or pca 
slices = np.empty((nr_parts,slices_per_part,new_size*new_size)) 
for dataset in data.datasets.values():
    for nr_images,instance in enumerate(dataset.instances):
        img = torch.tensor(instance.y.numpy())
        img = trans.resize_3d(img, size=(1,new_size,new_slices,new_size), label=True)
        img = img.numpy()
        for j in range(new_slices):
            part = math.floor(j/slices_per_image_part)
            i = j % slices_per_image_part + slices_per_image_part*nr_images
            slices[part,i]=img[:,:,j,:].flatten()


## prepare partitioning into parts
pr_slices_per_part = math.floor(20*new_slices*(1/nr_parts))
pr_slices_per_image_part = math.floor(new_slices/nr_parts) ##maybe change partitioning later 

# get segmented images in order to compare
pr_slices = np.empty((nr_parts,pr_slices_per_part,new_size*new_size)) 
for id,j in enumerate(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']):
    segmented_path = PATH_TO_NEW_SEGMENTATION
    path=os.path.join(segmented_path, 'segmented_lung_'+ j + '_gt.nii.gz')
    img = torchio.Image(path, type=torchio.LABEL)
    img = img.numpy()
    img = torch.tensor(img)
    img = trans.resize_3d(img, size=(1,new_size,new_slices,new_size), label=True)
    img = img.numpy()
    for j in range(new_slices):
        if np.sum(img[:,:,j,:]) == 0:
            pr_number_of_dark_slices = pr_number_of_dark_slices + 1
        part = math.floor(j/pr_slices_per_image_part)
        i = j % pr_slices_per_image_part + pr_slices_per_image_part*nr_images
        pr_slices[part,i]=img[:,:,j,:].flatten()

## prepare Visualisation  
if PLOT_RESULTS:
    fig, axs = plt.subplots(1,nr_parts)

#Get decomposition of data and plot it if wanted 
for part in range(nr_parts):
    if TOOL_FOR_DIM_RECTION == 'TrunctatedSVD':
        dim_red_tool=TruncatedSVD(random_state=42)
    else: 
        dim_red_tool=PCA(n_components=2,random_state=42)
    #transform the original data
    transformed_original_data = dim_red_tool.fit_transform(slices[part])
    for instance in range(8):
        lower_index = instance*slices_per_image_part
        upper_index = (instance+1)*slices_per_image_part
        if PLOT_RESULTS:
            axs[part].scatter(transformed_original_data[lower_index:upper_index,0],transformed_original_data[lower_index:upper_index,1],label=str(instance),marker="o")
    ## transform prostate Data with dim_red_tool
    transformed_new_segmentation = dim_red_tool.transform(pr_slices[part])
    for instance in range(8):
        pr_lower_index = instance*pr_slices_per_image_part
        pr_upper_index = (instance+1)*pr_slices_per_image_part
        if PLOT_RESULTS:
            axs[part].scatter(transformed_new_segmentation[pr_lower_index:pr_upper_index,0],transformed_new_segmentation[pr_lower_index:pr_upper_index,1],label=str(instance),marker="^")
if PLOT_RESULTS:
    plt.show()

## 7. Compute Number of #lonely Pixels 

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[0]


    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def lonely_pixels(image,threshold=2,neighbourhood=1,verbose=False):
    r'''computes the amount of pixels with label 1, who have a maximum of _threshhold_ neighbours with label 1 in the _neighbourhood_
    neighbourhood = "radius" of the kernel
    '''
    ## generate the kernel
    kernel_size = 1+2*neighbourhood
    kernel = np.ones((kernel_size,kernel_size))
    kernel_center_value = ((-1)*kernel_size*kernel_size)+1
    kernel[neighbourhood,neighbourhood]= kernel_center_value
    ##compute convolution
    padding=neighbourhood
    slide = convolve2D(image, kernel, padding=padding)

    ## compute threshold for sum
    neg_threshold = kernel_center_value + threshold 
    sum = 0
    for x in range(slide.shape[0]):
        for y in range(slide.shape[1]):
            if slide[x,y] <= neg_threshold:
                sum = sum + 1
    if verbose:
        print('There are {} pixels with label 1, who have a maximum of {} other pixels with label 1 in a neighbourhood  with radius {}'.format(sum,threshold,neighbourhood))
    return sum

# load both datasets and compute the number of lonely pixels per full image and compare
