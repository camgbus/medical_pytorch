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
from sklearn.decomposition import TruncatedSVD,PCA
import matplotlib.pyplot as plt 
import pickle 
import time 

# Hyperparams
MAKE_EXPERIMENT = True
ITERATIONS = [0]
RANDOM_STATES = [34,90789]
PATH_TO_DATA_STATISTICS = os.path.join('storage','statistics','UK_Frankfurt2','dim_red_exp_bbox_smaller_int_pca') #for saving the models/vectors
USE_ARPACK = False
GET_COMP_INFOS = True

start_time = time.time()

if MAKE_EXPERIMENT:
    # 2. reduce dimension of images to (57,256,256) and load components into an array 
    comp_infos = []
    seg_comp = []
    for i,dir in enumerate(os.listdir(os.path.join('downloads','UK_Frankfurt2'))):
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
            if GET_COMP_INFOS: 
                # save informations on components
                area = props[comp].area
                centroid = np.around(props[comp].centroid)
                convex_area = props[comp].convex_area
                convexity = np.around(area/convex_area,3)
                min_ax_length = round(props[comp].minor_axis_length)
                max_ax_length = round(props[comp].major_axis_length)
                informations = [i,comp,area,centroid,convexity,min_ax_length,max_ax_length]
                comp_infos.append(informations)
            min_row, min_col, min_sl, max_row, max_col, max_sl,  = props[comp].bbox
            part_of_seg = props[comp].image
            cut_seg = img[min_row:max_row,min_col:max_col,min_sl:max_sl]
            cut_shape = np.shape(cut_seg)
            for x in range(0,cut_shape[0]):
                for y in range(0,cut_shape[1]):
                    for z in range(0,cut_shape[2]):
                        if not part_of_seg[x,y,z]: #if this part of the bbox is not part of the segmentation, color is black -1024
                            cut_seg[x,y,z] = -1024                            
            cut_seg = torch.tensor(cut_seg).unsqueeze(0)
            cut_seg= resize_3d(cut_seg, size=(1,20,20,8),label=True)
            cut_seg = cut_seg.numpy()[0]
            cut_seg = cut_seg.flatten()
            seg_comp.append(cut_seg)
            comp += 1

    seg_comp = np.array(seg_comp)
    seg_comp_save = seg_comp
    if GET_COMP_INFOS:
        pickle.dump(comp_infos,open(os.path.join('storage','statistics','UK_Frankfurt2','UK_Frankfurt2_com_infos.sav'),'wb'))  
    print('Data Matrix has shape {}'.format(np.shape(seg_comp)))
    print('Beginning with transformations at {}'.format(time.time()))

    # repeat experiments mutiple times, with different random seeds :
    plts = 0 
    for rand_st in RANDOM_STATES:
        print('Testing random state {}'.format(rand_st))
        for nr_iterations in ITERATIONS:
            plts += 1
            print('Testing nr_iterations {} at time {}'.format(nr_iterations,time.time()))
            # 3. transform fit this array into 2 dim 
            seg_comp = seg_comp_save
            transformer = PCA(n_components=2)
            seg_comp = transformer.fit_transform(seg_comp)
            if rand_st == 34 and nr_iterations==10 :
                print('Transformed Data has shape {}'.format(np.shape(seg_comp)))

            # 4. plot and save
            plt.figure(plts)
            plt.scatter(seg_comp[:,0],seg_comp[:,1])
            plt.savefig(os.path.join(PATH_TO_DATA_STATISTICS,'dim_reduced_UK_Frankfurt2_rs{}_iters{}.png'.format(rand_st,nr_iterations)))
            pickle.dump(transformer,open(os.path.join(PATH_TO_DATA_STATISTICS,'transformer_UK_Frankfurt2_rs{}_iters{}.sav'.format(rand_st,nr_iterations)),'wb'))
            pickle.dump(seg_comp,open(os.path.join(PATH_TO_DATA_STATISTICS,'trans_data_UK_Frankfurt2_rs{}_iters{}.sav'.format(rand_st,nr_iterations)),'wb'))

    if USE_ARPACK:
        print('Now test with arpack at time {}'.format(time.time()))
        seg_comp = seg_comp_save
        transformer = TruncatedSVD(algorithm='arpack',random_state=42,tol=0.0001)
        seg_comp = transformer.fit_transform(seg_comp)
        #plot and save 
        plts += 1
        plt.figure(plts)
        plt.scatter(seg_comp[:,0],seg_comp[:,1])
        plt.savefig(os.path.join(PATH_TO_DATA_STATISTICS,'dim_reduced_UK_Frankfurt2_arpack.png'))
        pickle.dump(transformer,open(os.path.join(PATH_TO_DATA_STATISTICS,'transformer_UK_Frankfurt2_arpack.sav'),'wb'))
        pickle.dump(seg_comp,open(os.path.join(PATH_TO_DATA_STATISTICS,'trans_data_UK_Frankfurt2_arpack.sav'),'wb'))
else:  
    print('Loading experiment')
    transformer = pickle.load(open(os.path.join('storage','statistics','UK_Frankfurt2','Dim_Red_Experiments','transformer_UK_Frankfurt2_arpack.sav'),'rb'))
    reduced_seg_comp = pickle.load(open(os.path.join('storage','statistics','UK_Frankfurt2','Dim_Red_Experiments','trans_data_UK_Frankfurt2_arpack.sav'),'rb'))
    reduced_seg_comp_int = pickle.load(open(os.path.join('storage','statistics','UK_Frankfurt2','Dim_Red_Experiments_Intensity','trans_data_UK_Frankfurt2_rs34_iters10.sav'),'rb'))
    reduced_seg_comp_bbox = pickle.load(open(os.path.join('storage','statistics','UK_Frankfurt2','dim_red_exp_bbox','trans_data_UK_Frankfurt2_rs45_iters10.sav'),'rb'))
    comp_infos = pickle.load(open(os.path.join('storage','statistics','UK_Frankfurt2','UK_Frankfurt2_com_infos.sav'),'rb'))

    # visuelle Cluster in gruppen einteilen
    group00 = []
    group10 = []
    group01 = []
    for i,(x,y) in enumerate(reduced_seg_comp):
        if x<1 and y<1:
            group00.append(i)
        if x>1 and y<1:
            group10.append(i)
        if x<1 and y>1:
            group01.append(i)

    
    for el in group01:
        plt.scatter(reduced_seg_comp_bbox[el,0],reduced_seg_comp_bbox[el,1],color='green')
    for el in group10:
        plt.scatter(reduced_seg_comp_bbox[el,0],reduced_seg_comp_bbox[el,1],color='red')
    for el in group00:
        plt.scatter(reduced_seg_comp_bbox[el,0],reduced_seg_comp_bbox[el,1],color='blue')
    plt.show()

    # print('group00')
    # for i in group00:
    #     print(comp_infos[i])

    # print('\n')
    # print('\n')
    # print('group10')
    # for i in group10:
    #     print(comp_infos[i])

    # print('\n')
    # print('\n')
    # print('group01')
    # for i in group01:
    #     print(comp_infos[i])

    # rand_img = []
    # for i in range(5):
    #     img = np.random.random_integers(-1024,2000,(57,256,256))
    #     img = img.flatten()
    #     rand_img.append(img)
    # rand_img = np.array(rand_img)
    # reduced_rand_img = transformer.transform(rand_img)
    # plt.scatter(reduced_rand_img[:,0],reduced_rand_img[:,1],c='black')
    # plt.show()



        



    

















