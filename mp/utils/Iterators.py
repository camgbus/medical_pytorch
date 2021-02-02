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


class Dataset_Iterator():
    '''in order to iterate ove a dataset
    '''

    def __init__(self,data_path,mode='UK_Frankfurt2',resize=False,size=(1,256,256,57),size_components=(1,20,20,8)):

        self.data_path = data_path
        self.mode = mode 
        self.resize = resize 
        self.size = size 
        self.size_components = size_components
    
    def iterate_images(self,func,**kwargs):
        print('Starting iteration over images')
        output=[]
        if self.resize:
            if self.mode == 'UK_Frankfurt2':
                for dir in os.listdir(self.data_path):
                    path = os.path.join(self.data_path,dir)
                    img_path = os.path.join(path,'image.nii.gz')
                    seg_path = os.path.join(path,'mask.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())
                    img = resize_3d(img, size=(1,256,256,57))
                    seg = resize_3d(seg, size=(1,256,256,57), label=True)
                    img = img.numpy()[0]
                    seg = seg.numpy()[0]
                    values = func(img,seg,**kwargs)
                    output.append(values)

        else:
            if self.mode == 'UK_Frankfurt2':
                for dir in os.listdir(self.data_path):
                    path = os.path.join(self.data_path,dir)
                    img_path = os.path.join(path,'image.nii.gz')
                    seg_path = os.path.join(path,'mask.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    values = func(img,seg,**kwargs)
                    output.append(values)
        return output
    
    def iterate_components(self,func,threshold=10,**kwargs):
        print('Starting iteration over components')
        output=[]
        if self.resize:
            print('WARNING: func has to do the resizing alone, but gets parameter size_components')
            if self.mode == 'UK_Frankfurt2':
                for dir in os.listdir(self.data_path):
                    path = os.path.join(self.data_path,dir)
                    img_path = os.path.join(path,'image.nii.gz')
                    seg_path = os.path.join(path,'mask.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    labeled_image, nr_components = label(seg, return_num=True)
                    props = regionprops(labeled_image)
                    nr_components = len(props)
                    comp = 0
                    while comp < nr_components and props[comp].area > threshold:
                        output.append(func(img,seg,props[comp],self.size_components,**kwargs))
                        comp += 1      
        else:
            if self.mode == 'UK_Frankfurt2':
                for dir in os.listdir(self.data_path):
                    path = os.path.join(self.data_path,dir)
                    img_path = os.path.join(path,'image.nii.gz')
                    seg_path = os.path.join(path,'mask.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    labeled_image, nr_components = label(seg, return_num=True)
                    props = regionprops(labeled_image)
                    nr_components = len(props)
                    comp = 0
                    while comp < nr_components and props[comp].area > threshold:
                        output.append(func(img,seg,props[comp],**kwargs))
                        comp += 1                                        
        return output


class Component_Iterator():
    ''' In order to iterate over the components segmentation using the intensity image
    '''
    def __init__(self,img,seg,threshold=10):
        self.img = img 
        self.seg = seg
        self.threshold = threshold
    
    def iterate(self,func,**kwargs):
        values = []
        labeled_image, nr_components = label(self.seg, return_num=True)
        props = regionprops(labeled_image)
        nr_components = len(props)
        comp = 0
        while comp < nr_components and props[comp].area > self.threshold:
            values.append(func(self.img,self.seg,props[comp],**kwargs))
            comp += 1
        return values 