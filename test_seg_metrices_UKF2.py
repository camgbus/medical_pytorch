
# 1. Imports
from mp.data.datasets.ds_ct_UKF2 import UKF2
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

# 2. Hyperparameter 
USE_SERVER = False
CUDA_DEVICE = 1 
EPOCHS_TO_USE = [5,10]
PATH_TO_IMAGES = os.path.join('storage','data','UKF2')
PATH_TO_STATES = os.path.join('storage', 'exp', 'UKF2_seg_metrices', '0', 'states')
PATH_TO_NEW_SEGMENTATION = os.path.join('storage','data','UKF2_generated_seg')

# 3.Load segmentation agent from state and load data
if USE_SERVER:
    # device to use for computation config['device']
    device = 'cuda:'+str(CUDA_DEVICE)
else:
    device = torch.device('cpu')

# fill in information for segmentation model
# input shape of segmenting model config['input_shape']
input_shape = (1, 256, 256)
label_names = ['lung', 'consolidation']
nr_labels = 2  # number of labels in data

# load model
model = UNet2D(input_shape, nr_labels)
model.to(device)
agent = SegmentationAgent(model=model, label_names=label_names, device=device)

# load data on which decomposition is trained and which contains ground truth
data = Data()
data.add_dataset(UKF2())

# 4. Segment the images using the models from the given epochs
if not os.path.isdir(PATH_TO_NEW_SEGMENTATION):
    os.makedirs(PATH_TO_NEW_SEGMENTATION)

for epoch in EPOCHS_TO_USE:
    path_seg_epoch = os.path.join(PATH_TO_NEW_SEGMENTATION,'epoch_{}'.format(epoch))
    if not os.path.isdir(path_seg_epoch):
        print('Segmenting Images for epoch {}'.format(epoch))
        os.makedirs(path_seg_epoch)        
        agent.restore_state(PATH_TO_STATES, 'epoch_{}'.format(epoch))
        names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name in os.listdir(PATH_TO_IMAGES))
        for id,name in enumerate(names):
            #Ausrichtung muss noch geprüft werden !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            x_path = os.path.join(PATH_TO_IMAGES, name + '.nii.gz')
            x = torch.tensor(torchio.Image(x_path, type=torchio.INTENSITY).numpy())
            original_size_2d = x.shape[:3]
            original_size = x.shape
            x = x.permute(3, 0, 1, 2)
            pred = []
            with torch.no_grad():
                for slice_idx in range(len(x)):
                    inputs = trans.resize_2d(
                        x[slice_idx], size=(1, 256, 256)).to(agent.device)
                    inputs = torch.unsqueeze(inputs, 0)
                    slice_pred = agent.predict(inputs).float()
                    pred.append(trans.resize_2d(
                        slice_pred, size=original_size_2d, label=True))
            # Merge slices and rotate so depth last
            pred = torch.stack(pred, dim=0)  # depth,channel,weight,height
            pred = pred.permute(1, 2, 3, 0)  # ? channel,weight,height,depth is that right ? 
            assert original_size == pred.shape
            pred = pred.numpy()
            shape = pred.shape
            pred = np.resize(pred, (shape[1], shape[2], shape[3]))
            sitk.WriteImage(sitk.GetImageFromArray(pred), os.path.join(
                PATH_TO_NEW_SEGMENTATION, 'segmented_lung_' + str(id) + '_gt.nii.gz'))
        print('Images segmented and saved')
        

