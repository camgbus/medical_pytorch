
# 1. Imports
from mp.data.datasets.corona_fra_seg import UKF2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.data.data import Data
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.agents.segmentation_agent import SegmentationAgent

from scipy.stats import spearmanr
import os
import SimpleITK as sitk
import numpy as np
import torchio
import pickle

from sklearn.neural_network import MLPRegressor

import mp.data.pytorch.transformation as trans

import matplotlib.pyplot as plt

from compute_metrics_on_segmentation import compute_metrics

# 2. Hyperparameter 
USED_METRIC = 'avg_kl_dice_comp'
DESCRIPTION = '''Take the average kl_div of the histograms from the estimated density. Not using clustering and density is estimated by gaussian kernel with bw 20
                        Also take the avg dice of the consolidations and the avg dice differences as well as number of connected components'''
USE_SERVER = False
CUDA_DEVICE = 0 
PLOT_AVG_VS_DICE = True
RESIZED = True 
EPOCHS_TO_USE = [1,2,3,4,5,6,7,8,9]
IMG_TO_TEST = 5

PATH_TO_STATES = os.path.join('storage', 'models', 'UNet2D','UKF2','states')
PATH_TO_NEW_SEGMENTATION = os.path.join('storage','data','UKF2_generated_seg')
PATH_TO_SCORES = os.path.join('storage','statistics','UKF2','tests_seg_metrices')
if RESIZED:
    PATH_TO_IMAGES = os.path.join('storage','data','UKF2_resized')
else: 
    PATH_TO_IMAGES = os.path.join('storage','data','UKF2')


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
def get_dice(name,epoch):
    ''' computes the dice scores between the seg for img with name and its segmentations from the models using the given epochs
    '''
    seg_path = os.path.join(PATH_TO_IMAGES,name+'_gt.nii.gz')
    seg = sitk.ReadImage(seg_path)
    seg = sitk.GetArrayFromImage(seg)
    seg = seg.flatten()
    gen_seg_path = os.path.join(PATH_TO_NEW_SEGMENTATION,'epoch_{}'.format(epoch),name+'_gt.nii.gz')
    gen_seg = sitk.ReadImage(gen_seg_path)
    gen_seg = sitk.GetArrayFromImage(gen_seg)
    gen_seg = gen_seg.flatten()
    dice_score = np.dot(seg,gen_seg)
    return dice_score

def get_metric(name,epoch):
    img_path = os.path.join(PATH_TO_IMAGES,name+'.nii.gz')
    seg_path = os.path.join(PATH_TO_NEW_SEGMENTATION,'epoch_{}'.format(epoch),name+'_gt.nii.gz')
    hist_score,_,dice,comp = compute_metrics(img_path,seg_path)
    return [hist_score,dice,comp]

def get_prediction(img_path,name,save_path):
    x = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())
    original_size_2d = x.shape[:3]
    original_size = x.shape
    x = x.permute(3, 0, 1, 2)
    pred = []
    with torch.no_grad():
        for slice_idx in range(len(x)):
            if not RESIZED:
                inputs = trans.resize_2d(x[slice_idx], size=(1, 256, 256))                     
            inputs = torch.unsqueeze(inputs, 0).to(agent.device)
            slice_pred = agent.predict(inputs).float()
            if not RESIZED:
                slice_pred = trans.resize_2d(
                slice_pred, size=original_size_2d, label=True)
            pred.append(slice_pred)
    # Merge slices and rotate so depth last
    pred = torch.stack(pred, dim=0)  # depth,channel,weight,height
    pred = pred.permute(1, 2, 3, 0)  # ? channel,weight,height,depth is that right ? 
    assert original_size == pred.shape
    pred = pred.numpy()
    shape = pred.shape
    pred = np.resize(pred, (shape[1], shape[2], shape[3]))
    sitk.WriteImage(sitk.GetImageFromArray(pred), os.path.join(
        save_path, name + '_gt.nii.gz'))

def flatten_list(alist):
    return_list = []
    for entry in alist:
        if type(entry) is not list:
            return_list.append(entry)
        else: 
            for ele in entry:
                return_list.append(ele)
    return return_list

def l2_loss(pred,truth):
    result = np.sum(np.power((pred-truth),2))
    return result

def get_l2_loss(X_test,y_test,predictor):
    y_pred = predictor.predict(X_test)
    loss = l2_loss(y_pred,y_test)
    return loss 

names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name in os.listdir(PATH_TO_IMAGES))

# 4. Segment the images using the models from the given epochs
if not os.path.isdir(PATH_TO_NEW_SEGMENTATION):
    os.makedirs(PATH_TO_NEW_SEGMENTATION)

for epoch in EPOCHS_TO_USE:
    path_seg_epoch = os.path.join(PATH_TO_NEW_SEGMENTATION,'epoch_{}'.format(epoch))
    if not os.path.isdir(path_seg_epoch):
        print('Segmenting Images for epoch {}'.format(epoch))
        os.makedirs(path_seg_epoch)        
        agent.restore_state(PATH_TO_STATES, 'epoch_{}'.format(epoch))
        for id,name in enumerate(names):
            x_path = os.path.join(PATH_TO_IMAGES, name + '.nii.gz')
            get_prediction(x_path,name,path_seg_epoch)
        print('Images segmented and saved')

# 5. get all the scores for every desired epoch 

for epoch in EPOCHS_TO_USE:
    #dice scores
    path_to_dice = os.path.join(PATH_TO_SCORES,'dice','dice_epoch{}.npy'.format(epoch))
    if not os.path.isfile(os.path.join(path_to_dice)):
        dice_scores = []
        for name in names:
            dice_scores.append(get_dice(name,epoch))    
        pickle.dump(open(path_to_dice,'wb'),dice_scores)

    path_to_metric = os.path.join(PATH_TO_SCORES,USED_METRIC,'metrics_epoch_{}.npy'.format(epoch))
    if not os.path.isfile(os.path.join(path_to_metric)):
        metric_scores = []
        for name in names:
            metric_scores.append(get_metric(name,epoch))
        pickle.dump(open(path_to_metric,'wb'),metric_scores)

describtion_name = os.path.join(PATH_TO_SCORES,USED_METRIC,'metric_describtion.txt')
if not os.path.isfile(describtion_name):
    descr = open(describtion_name,'w')
    descr.write(DESCRIPTION)
    descr.close()

#6. get dice and metrices into an array and directly split into train and test set
#      i want to use the same images to test in order to prevent data leakage 
dice_scores_train = []
dice_scores_test = []
metric_scores_train = []
metric_scores_test = []

for epoch in EPOCHS_TO_USE:
    path_to_metric = os.path.join(PATH_TO_SCORES,USED_METRIC,'metrics_epoch_{}.npy'.format(epoch))
    path_to_dice = os.path.join(PATH_TO_SCORES,'dice','dice_epoch{}.npy'.format(epoch))
    dice_list_epoch = pickle.load(open(path_to_dice,'rb'))
    metric_list_epoch = pickle.load(open(path_to_metric,'rb'))
    flattened_scores = [flatten_list(entry) for entry in metric_list_epoch]

    dice_scores_train += dice_list_epoch[:IMG_TO_TEST]
    dice_scores_test += dice_list_epoch[IMG_TO_TEST:]
    metric_scores_train += flattened_scores[:IMG_TO_TEST]
    metric_scores_test += flattened_scores[IMG_TO_TEST:]

X_train = np.array(metric_scores_train)
X_test = np.array(metric_scores_test)
y_train = np.array(dice_scores_train)
y_test = np.array(dice_scores_test)

# 7.get visual confirmation
if PLOT_AVG_VS_DICE:
    dice = np.append(y_train,y_test)
    avg_dist = [ele[0] for ele in metric_scores_test+metric_scores_train]
    corr,pval = spearmanr(dice,avg_dist)
    print('The correlation between the two values is {}'.format(corr))

    plt.scatter(dice,avg_dist)
    plt.show()

# 8. train a NN to predict dice from metrices
# A list of setting for the MLP to be trained with 

settings = [{'name':'first','size':(100,100,100,60,30,10),'lr':'adaptive','solver':'adam'},
            {'name':'smaller','size':(50,50,30,10),'lr':'adaptive','solver':'adam'},
            {'name':'very_smal','size':(20,20),'lr':'adaptive','solver':'adam'}]

for setting in settings:
    regressor_name = setting['name']
    path_regr = os.path.join(PATH_TO_SCORES,USED_METRIC,regressor_name+'regressor','regression_model.sav')
    regr_descr_path = os.path.join(PATH_TO_SCORES,USED_METRIC,regressor_name+'regressor','describtion.txt')
    size = setting['size']
    lr = setting['lr']
    solver = setting['solver']

    if os.path.isfile(path_regr):
        regressor = pickle.load(open(path_regr,'rb'))  
    else:
        regressor = MLPRegressor(size,learning_rate=lr,random_state=1,verbose=False,solver=solver )
        regressor.fit(X_train,y_train)

        regressor_score = regressor.score(X_test,y_test)
        l2_loss = get_l2_loss(X_test,y_test,regressor)
        losses_string = 'The regressor {} has a score of {} and an l2 loss of {}'.format(regressor_name,regressor_score,l2_loss)
        print()

        with open(path_regr,'wb') as saver:
            pickle.dump(regressor,saver)
        
        regression_descr = r"size={},learning_rate={},solver={}".format(size,lr,solver)+ "used epochs are {}".format(EPOCHS_TO_USE) + losses_string
        descr = open(regr_descr_path,'w')
        descr.write(regression_descr)
        descr.close()




    

