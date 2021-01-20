
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
from scipy.stats import spearmanr

import os
import SimpleITK as sitk
import numpy as np
import torchio
import math

from mp.eval.inference.predictor import Predictor2D
from mp.eval.inference.predictor import Predictor3D

import mp.data.pytorch.transformation as trans

import matplotlib.pyplot as plt

from compute_metrics_on_segmentation import compute_metrics

# 2. Hyperparameter 
USED_METRIC = 'average_kl_div_of_hist'
DESCRIPTION = 'Take the average kl_div of the histograms from the estimated density. Not using clustering and density is estimated by gaussian kernel with bw 20'
USE_SERVER = False
CUDA_DEVICE = 1 
EPOCHS_TO_USE = [5,10]
PATH_TO_IMAGES = os.path.join('storage','data','UKF2')
PATH_TO_STATES = os.path.join('storage', 'exp', 'UKF2_seg_metrices', '0', 'states')
PATH_TO_NEW_SEGMENTATION = os.path.join('storage','data','UKF2_generated_seg')
PATH_TO_SCORES = os.path.join('storage','statistics','UK_Frankfurt2','tests_seg_metrices')


if not os.path.isdir(PATH_TO_SCORES):
    os.makedirs(PATH_TO_SCORES)
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

# 3.1 get some helper functions
def get_dice(name,epochs):
    ''' computes the dice scores between the seg for img with name and its segmentations from the models using the given epochs
    '''
    seg_path = os.path.join(PATH_TO_IMAGES,name+'_gt.nii.gz')
    seg = sitk.ReadImage(seg_path)
    seg = sitk.GetArrayFromImage(seg)
    seg = seg.flatten()
    dices_scores = []
    for epoch in epochs:
        gen_seg_path = os.path.join(PATH_TO_NEW_SEGMENTATION,'epoch_{}'.format(epoch),name+'_gt.nii.gz')
        gen_seg = sitk.ReadImage(gen_seg_path)
        gen_seg = sitk.GetArrayFromImage(gen_seg)
        gen_seg = gen_seg.flatten()
        dice_score = np.dot(seg,gen_seg)
        dice_scores.append(dice_score)
    return dices_scores

def get_metric(name,epochs):
    img_path = os.path.join(PATH_TO_IMAGES,name+'.nii.gz')
    hist_scores = []
    for epoch in epochs:
        seg_path = os.path.join(PATH_TO_NEW_SEGMENTATION,'epoch_{}'.format(epoch),name+'_gt.nii.gz')
        hist_score,_,_,_ = compute_metrics(img_path,seg_path)
        hist_scores.append(hist_score)
    return hist_scores




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
                path_seg_epoch, name + '_gt.nii.gz'))
        print('Images segmented and saved')

# 5. get all the scores 
names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name in os.listdir(PATH_TO_IMAGES))


for epoch in EPOCHS_TO_USE:
    #dice scores
    path_to_dice = os.path.join(PATH_TO_SCORES,'dice_epoch{}.npy'.format(epoch))
    if not os.path.isfile(os.path.join(path_to_dice)):
        dice_scores = []
        for name in names:
            dice_scores.append(get_dice(name,[epoch]))    
        np.save(path_to_dice,np.array(dice_scores))

    path_to_metric = os.path.join(PATH_TO_SCORES,USED_METRIC+'epoch_{}.npy'.format(epoch))
    if not os.path.isfile(os.path.join(path_to_metric)):
        metric_scores = []
        for name in names:
            metric_scores.append(get_metric(name,[epoch]))
        np.save(path_to_metric,np.array(metric_scores))

describtion_name = os.path.join(PATH_TO_SCORES,USED_METRIC+'_descr.txt')
if not os.path.isfile(describtion_name):
    descr = open(describtion_name,'w')
    descr.write(DESCRIPTION)
    descr.close()



#6. compute the ranks of the scores, take care of reverse order 
corr,pval = spearmanr(dice_scores,hist_scores)
print('The Spearman Correlation between the two scores is {}'.format(corr))

plt.scatter(dice_scores,hist_scores)
plt.show()




