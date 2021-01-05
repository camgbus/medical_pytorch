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
        component_mask = np.zeros(shape)
        for x,y,z in coords: 
            component_mask[x,y,z] = 1
        component_mask = component_mask.flatten()
        seg_comp.append(component_mask)
        comp += 1

print('Beginning with transformation')

# 3. transform fit this array into 2 dim 
transformer = TruncatedSVD(n_iter=10,random_state=34)
seg_comp = transformer.fit_transform(seg_comp)

# 4. plot and save 
print('Plotting and saving')
plt.scatter(seg_comp[:,0],seg_comp[:,1])
plt.savefig(os.path.join('storage','statistics','UK_Frankfurt2','dim_reduced_UK_Frankfurt2.png'))
pickle.dump(transformer,open(os.path.join('storage','statistics','UK_Frankfurt2','transformer_UK_Frankfurt2.sav'),'wb'))
pickle.dump(seg_comp,open(os.path.join('storage','statistics','UK_Frankfurt2','trans_data_UK_Frankfurt2.sav'),'wb'))














