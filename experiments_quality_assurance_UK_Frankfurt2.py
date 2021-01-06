## Dimensionality reduction and visualisation of each segmented region 
# First of all, the images are resized to the same dimesionality (?,256,256)
# Then, each component, that is big enough (we start with size>500), is taken and we feed all of them into the dim. Reduction.

# 1. Imports 
import os 
import SimpleITK as sitk 
import numpy as np 
import torchio
import torch
from mp.data.pytorch.transformation import resize_3d
from skimage.measure import label,regionprops
from torch.nn.functional import interpolate
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt 
import pickle 
import time 

# Hyperparams
MAKE_EXPERIMENT = True 
PATH_TO_DATA_STATISTICS = os.path.join('storage','statistics','UK_Frankfurt2')

start_time = time.time()

if MAKE_EXPERIMENT:
    # 2. reduce dimension of images to (57,256,256) and load components into an array 
    seg_comp = []
    for dir in os.listdir(os.path.join('downloads','UK_Frankfurt2')):
        path = os.path.join('downloads','UK_Frankfurt2',dir)
        img_path = os.path.join(path,'image.nii.gz')
        seg_path = os.path.join(path,'mask.nii.gz')
        img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())
        seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())
        
        #2.1 resize 
        img = resize_3d(img, size=(1,256,256,57))
        seg = resize_3d(seg, size=(1,256,256,57), label=True)
        img = img.numpy()[0]
        seg = seg.numpy()[0]
        
        shape = np.shape(seg)
        components = label(seg)
        props = regionprops(components,img)
        props = sorted(props ,reverse=True, key =lambda dict:dict['area'])
        number_components = len(props)

        # 2.2 read out the components
        comp = 0
        while comp < number_components and props[comp].area > 500 : 
            coords = props[comp].coords
            component_mask = np.full(shape,-1024,dtype=int)
            for x,y,z in coords: 
                component_mask[x,y,z] = int(img[x,y,z])
            component_mask = component_mask.flatten()
            seg_comp.append(component_mask)
            comp += 1

    seg_comp = np.array(seg_comp)
    seg_comp_save = seg_comp
    print('Data Matrix has shape {}'.format(np.shape(seg_comp)))
    print('Beginning with transformations at {}'.format(time.time()))

    # repeat experiments mutiple times, with different random seeds :
    plts = 0 
    for rand_st in [34,3243242]:
        print('Testing random state {}'.format(rand_st))
        for nr_iterations in [10,30]:
            plts += 1
            print('Testing nr_iterations {} at time {}'.format(nr_iterations,time.time()))
            # 3. transform fit this array into 2 dim 
            seg_comp = seg_comp_save
            transformer = TruncatedSVD(n_iter=nr_iterations,random_state=rand_st)
            seg_comp = transformer.fit_transform(seg_comp)
            if rand_st == 34 and nr_iterations==10 :
                print('Transformed Data has shape {}'.format(np.shape(seg_comp)))

            # 4. plot and save
            plt.figure(plts)
            plt.scatter(seg_comp[:,0],seg_comp[:,1])
            plt.savefig(os.path.join(PATH_TO_DATA_STATISTICS,'Dim_Red_Experiments_Intensity','dim_reduced_UK_Frankfurt2_rs{}_iters{}.png'.format(rand_st,nr_iterations)))
            pickle.dump(transformer,open(os.path.join(PATH_TO_DATA_STATISTICS,'Dim_Red_Experiments_Intensity','transformer_UK_Frankfurt2_rs{}_iters{}.sav'.format(rand_st,nr_iterations)),'wb'))
            pickle.dump(seg_comp,open(os.path.join(PATH_TO_DATA_STATISTICS,'Dim_Red_Experiments_Intensity','trans_data_UK_Frankfurt2_rs{}_iters{}.sav'.format(rand_st,nr_iterations)),'wb'))

    # Use arpack 
    print('Now test with arpack at time {}'.format(time.time()))
    seg_comp = seg_comp_save
    transformer = TruncatedSVD(algorithm='arpack',random_state=42,tol=0.0001)
    seg_comp = transformer.fit_transform(seg_comp)

    #plot and save arpack
    plts += 1
    plt.figure(plts)
    plt.scatter(seg_comp[:,0],seg_comp[:,1])
    plt.savefig(os.path.join(PATH_TO_DATA_STATISTICS,'Dim_Red_Experiments_Intensity','dim_reduced_UK_Frankfurt2_arpack.png'))
    pickle.dump(transformer,open(os.path.join(PATH_TO_DATA_STATISTICS,'Dim_Red_Experiments_Intensity','transformer_UK_Frankfurt2_arpack.sav'),'wb'))
    pickle.dump(seg_comp,open(os.path.join(PATH_TO_DATA_STATISTICS,'Dim_Red_Experiments_Intensity','trans_data_UK_Frankfurt2_arpack.sav'),'wb'))
else:  
    print('Loading experiment')
    transformer = pickle.load(open(os.path.join(PATH_TO_DATA_STATISTICS,'Dim_Red_Experiments_Intensity','transformer_UK_Frankfurt2_arpack.sav'),'rb'))
    reduced_seg_comp = pickle.load(open(os.path.join(PATH_TO_DATA_STATISTICS,'Dim_Red_Experiments_Intensity','trans_data_UK_Frankfurt2_arpack.sav'),'rb'))
    print(len(reduced_seg_comp))


















