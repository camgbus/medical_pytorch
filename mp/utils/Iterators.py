import os 
import numpy as np 
import torchio
import torch
from skimage.measure import label,regionprops

def iterate_components(img,seg,func,output,threshold,**kwargs):
    '''Gets a pair of img, seg, and a function that should iterate over its components. 
    Output is the vec, that the results are appended to and threshhold gives a min size for 
    the components we would like to iterate over

    Args:
        img (ndarray): the image 
        seg (ndarray): its segmentation 
        func ((img,seg,props,**kwargs)->object): computes object for one component given by the regionprops 
            dict props for the img-seg pair. Should return a number or a list of numbers. 
        output (list(numbers)): the list the results are appended to
        threshhold (int): components have to be this big in order to get iterated over
    '''
    labeled_image, nr_components = label(seg, return_num=True)
    props = regionprops(labeled_image)
    props = sorted(props ,reverse=True, key =lambda dict:dict['area'])
    nr_components = len(props)
    comp = 0
    while comp < nr_components and props[comp].area > threshold:
        output.append(func(img,seg,props[comp],**kwargs))
        comp += 1 


class Dataset_Iterator():
    '''in order to iterate ove a dataset
    '''

    def __init__(self,data_path,mode='UK_Frankfurt2',resize=False,size=(1,256,256,57),size_components=(1,20,20,8)):
        '''A iterator, who iterates over the all images in a given folder

        Args:
            data_path (str): the path to the data
            mode (str): should be either:
                - UK_Frankfurt2: here the data has a structure, where every patient has its own id-dir
                    in which img.nii.gz and seg... can be found 
                - normal: all images and seg are in the same dir and can be easily iteratet over, they can be identified by 
                    their unique id and their endings id.nii.gz is img, whereas id_gt.nii.gz is a seg.
                - JIP : Here every patient has its own dir, containing a dir for img and a dir for seg in which only one img file 
                    with ending .mhd is.
            resize (bool): if the images should be reiszed, is not implemented
            size (tuple): the size, the imf should be resized to
            size_components (tuple): the size comp should be resized to
        '''
        self.data_path = data_path
        self.mode = mode 
        self.resize = resize 
        self.size = size 
        self.size_components = size_components
        self.jip_extension = 'mhd'
    
    def iterate_images(self,func,**kwargs):
        '''Iterates over all images in the given path,  and accumulates the results of func in a list, 
        that contains an object for every processed image-seg pair 
        
        Args: 
            func (img,seg,**kwargs)->object : a function that maps a image-seg pair to an object (e.g. some property)

        Returns (list(objects)): a list of objects, with every object corresponding to one image-segmentation pair
        '''
        print('Starting iteration over images')
        output=[]
        if self.resize:
            #code for this case is on the chArmin branch
            raise NotImplementedError

        else:
            if self.mode == 'UK_Frankfurt2':
                for dir in sorted(os.listdir(self.data_path)):
                    path = os.path.join(self.data_path,dir)
                    img_path = os.path.join(path,'image.nii.gz')
                    seg_path = os.path.join(path,'mask.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    values = func(img,seg,**kwargs)
                    output.append(values)

            if self.mode == 'normal':
                names = sorted(list(set(file_name.split('.nii')[0].split('_gt')[0] 
                            for file_name in os.listdir(self.data_path))))
                for name in names: 
                    seg_path = os.path.join(self.data_path,name+'_gt.nii.gz')
                    img_path = os.path.join(self.data_path,name+'.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    values = func(img,seg,**kwargs)
                    output.append(values)

            if self.mode == 'JIP':
                names = sorted(os.listdir(self.data_path))
                for name in names: 
                    seg_path = os.path.join(self.data_path,name,'seg','001.{}'.format(self.jip_extension))
                    img_path = os.path.join(self.data_path,name,'img','img.{}'.format(self.jip_extension))
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    values = func(img,seg,**kwargs)
                    output.append(values)
            
            if self.mode == 'JIP_test':
                names = sorted(os.listdir(self.data_path))
                for name in names: 
                    seg_path = os.path.join(self.data_path,name,'seg','001.nii.gz')
                    img_path = os.path.join(self.data_path,name,'img','img.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    values = func(img,seg,**kwargs)
                    output.append(values)
        return output
    
    def iterate_components(self,func,threshold=1000,**kwargs):
        '''Iterates over all connected components of all images in the given path, that are bigger then 
        threshhold and accumulates the results of func in a list, that contains an object for every con.comp. processed 
        
        Args: 
            func (img,seg,props,**kwargs)->object : a function that maps a connected component to an object (e.g. some property)

        Returns (list(objects)): a list of objects, with every object corresponding to one image-segmentation pair
        '''
        print('Starting iteration over components')
        output=[]
        if self.resize:
            raise NotImplementedError
        else:
            if self.mode == 'UK_Frankfurt2':
                for dir in sorted(os.listdir(self.data_path)):
                    path = os.path.join(self.data_path,dir)
                    img_path = os.path.join(path,'image.nii.gz')
                    seg_path = os.path.join(path,'mask.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    iterate_components(img,seg,func,output,threshold,**kwargs)

            if self.mode == 'normal':
                #get the names of the files, it is assumed, that the data has the endings for mask and img as UK_Frankfurt
                names = sorted(list(set(file_name.split('.nii')[0].split('_gt')[0] 
                            for file_name in os.listdir(self.data_path))))
                for name in names:
                    seg_path = os.path.join(self.data_path,name+'_gt.nii.gz')
                    img_path = os.path.join(self.data_path,name+'.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    iterate_components(img,seg,func,output,threshold,**kwargs) 
            
            if self.mode == 'JIP':
                names = sorted(os.listdir(self.data_path))
                for name in names: 
                    seg_path = os.path.join(self.data_path,name,'seg','001.{}'.format(self.jip_extension))
                    img_path = os.path.join(self.data_path,name,'img','img.{}'.format(self.jip_extension))
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    iterate_components(img,seg,func,output,threshold,**kwargs)
            
            if self.mode == 'JIP_test':
                names = sorted(os.listdir(self.data_path))
                for name in names: 
                    seg_path = os.path.join(self.data_path,name,'seg','001.nii.gz')
                    img_path = os.path.join(self.data_path,name,'img','img.nii.gz')
                    img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
                    seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
                    iterate_components(img,seg,func,output,threshold,**kwargs)
        return output


class Component_Iterator():
    ''' In order to iterate over the components segmentation using the intensity image
    '''
    def __init__(self,img,seg,threshold=1000):
        self.img = img 
        self.seg = seg
        self.threshold = threshold
    
    def iterate(self,func,**kwargs):
        '''Iterates over all connected components of the given img-seg pair, that are bigger then 
        threshhold and accumulates the results of func in a list, that contains an object for every con.comp. processed 
        
        Args: 
            func (img,seg,props,**kwargs)->object : a function that maps a connected component to an object (e.g. some property)

        Returns (list(objects)): a list of objects, with every object corresponding to one image-segmentation pair
        '''
        values = []
        iterate_components(self.img,self.seg,func,values,self.threshold,**kwargs)
        return values 