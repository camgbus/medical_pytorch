import os 
from mp.paths import JIP_dir
import numpy as np
import torchio
import torch
from skimage.measure import label,regionprops
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
#set environmental variables
#for data_dirs folder, nothing changed compared to Simons version 
os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
os.environ["OPERATOR_IN_DIR"] = "input"
os.environ["OPERATOR_OUT_DIR"] = "output"
os.environ["OPERATOR_TEMP_DIR"] = "temp"
os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')

# preprocessing dir and subfolders 
os.environ["PREPROCESSED_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'preprocessed_dirs')
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"] = "output_scaled"
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = "output_scaled_train"

#dir where train data for intensites is stored (this only needs to be trains_dirs, but since i have more 
# datasets, another subfolder is here)
os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')

#ignore
##below is for christian only, used for older data structures where models are trained on
os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('Covid-RACOON','All images and labels')
os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('Covid-RACOON','All predictions')
#os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('gt_small')
#os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('pred_small')

#which mode is active either 'train' or 'inference' 
os.environ["INFERENCE_OR_TRAIN"] = 'train'

#ignore
# the ending of the image files in train_dir is only for older datasets
os.environ["INPUT_FILE_ENDING"] = 'nii.gz'

from mp.utils.preprocess_utility_functions import basic_preprocessing
from mp.utils.preprocess_utility_functions import extract_features_all_data,compute_all_prediction_dice_scores
from train_restore_use_models.train_int_based_quantifier import train_dice_predictor
from mp.utils.intensities import sample_intensities
from mp.models.densities.density import Density_model
from mp.utils.feature_extractor import Feature_extractor

def filter_segmentations_dens(id):
    return id.startswith('Task541') or id.startswith('Task740')

def sample_filtered_intensities(filter):
    work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    output = []
    for id in os.listdir(work_path):
        if filter(id):
            seg_path = os.path.join(work_path,id,'seg','001.nii.gz')
            img_path = os.path.join(work_path,id,'img','img.nii.gz')
            # pylint : disable = not-callable
            seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
            img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
            labeled_image, nr_components = label(seg, return_num=True)
            props = regionprops(labeled_image)
            props = sorted(props ,reverse=True, key =lambda dict:dict['area'])
            nr_components = len(props)
            comp = 0
            while comp < min(nr_components,3) and props[comp].area > 100:
                output.append(sample_intensities(img,seg,props[comp],number=750))
                comp += 1 
    return output

def filter_feature_extr(id,model):
    # train data 
    if model[:7] in ['Task740'] and id[:7] in ['Task740','Task541']:
        return 'train'
    if model[:7] in ['Task089'] and id[:7] in ['Task541']:
        return 'train'
    # id data
    if model[:7] in ['Task740'] and id[:7] in ['Task741']:
        return 'gc_gc'
    if model[:7] in ['Task740'] and id[:7] in ['Task542']:
        return 'gc_frankfurt'
    # od data 
    if model[:7] in ['Task740'] and id[:7] in ['Task200','Task201']:
        return 'gc_mosmed'
    if model[:7] in ['Task740'] and id[:7] in ['Task100','Task101']:
        return 'gc_radiopedia'
    
    return 'other'

def extract_features_train_id_od(filter,splits):
    X = []
    y = []
    for i in range(len(splits)):
        X.append([])
        y.append([])
    dens = Density_model()
    feat_extr = Feature_extractor(dens)
    work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    for id in os.listdir(work_path):
        all_pred_path = os.path.join(work_path,id,'pred')
        if os.path.exists(all_pred_path):
            for model in os.listdir(all_pred_path):
                split = filter(id,model)
                for i,s in enumerate(splits):
                    if split == s:
                        feature_path = os.path.join(all_pred_path,model,'features.json')
                        label_path = os.path.join(all_pred_path,model,'dice_score.json')
                        feat_vec = feat_extr.read_feature_vector(feature_path)
                        # feat_vec = [feat_vec[1],feat_vec[3],feat_vec[4]]
                        label = feat_extr.read_prediction_label(label_path)
                        if np.isnan(np.sum(np.array(feat_vec))) or feat_vec[0]>100000:
                            pass 
                        else:
                            X[i].append(feat_vec)
                            y[i].append(label)
                            
    return X,y

def l2_loss(pred,truth):
    n = len(pred)
    return (1/n)*(np.sum((pred - truth)**2))

def l1_loss(pred,truth, std = False):
    losses = np.absolute(pred-truth)
    mean = np.mean(losses)
    std = np.std(losses)
    if std:
        return mean,std
    return mean

def l1_loss_bins(pred,truth,split,by_sign=True):
    save_path = os.path.join('storage','Results','Setup 5','Bins_{}'.format(split))
    if not by_sign:
        loss_by_bin = [[] for _ in range(10)]
        bins = np.array([])
        bins_std = np.array([])
        for i in range(10):
            for pre,tru in zip(pred,truth):
                if i*0.1 <= tru and (i+1)*0.1 > tru:
                    loss_by_bin[i].append(abs(pre-tru))
            if loss_by_bin[i]:
                print('for the bin [{:.1f},{:.1f}) the error is {:.3f} over {} data_points'.format(i*0.1,(i+1)*0.1,np.mean(loss_by_bin[i]),len(loss_by_bin[i])))
                bins = np.append(bins,np.mean(loss_by_bin[i]))
                bins_std = np.append(bins_std,np.std(loss_by_bin[i]))
            else:
                bins = np.append(bins,0)
                bins_std = np.append(bins_std,0)
        plt.bar(0.05+np.arange(10)*0.1,height = bins, width = 0.05,yerr = bins_std)
        plt.plot(0.05+np.arange(10)*0.1,[0.1 for i in range(10)],'-r')
        plt.title(split)
    else : 
        widths = [[] for _ in range(10)]
        loss_by_bin = [[[],[]] for _ in range(10)]
        bins = [[] for _ in range(10)]
        bins_std = [[] for _ in range(10)]
        for i in range(10):
            for pre,tru in zip(pred,truth):
                if i*0.1 <= tru and (i+1)*0.1 > tru:
                    err = pre-tru
                    if err >= 0:
                        loss_by_bin[i][1].append(err)
                    else:
                        loss_by_bin[i][0].append(err)
            for j in range(2):
                if loss_by_bin[i][j]:
                    bins[i].append(np.mean(loss_by_bin[i][j]))
                    bins_std[i].append(np.std(loss_by_bin[i][j]))
                else:
                    bins[i].append(0)
                    bins_std[i].append(0)
            unders = len(loss_by_bin[i][0])
            overs = len(loss_by_bin[i][1])
            total = unders+overs
            if total > 0:
                if unders/total > 0.66:
                    widths[i]=[0.05,0.03]
                elif overs/total >0.66:
                    widths[i]=[0.03,0.05]
                else: 
                    widths[i]=[0.05,0.05]
            else:
                widths[i]=[0.05,0.05]
        bins = np.array(bins)
        bins_std = np.array(bins_std)
        widths = np.array(widths)
        plt.bar(0.05+np.arange(10)*0.1,height = bins[:,0], width = widths[:,0],yerr = bins_std[:,0])
        plt.bar(0.05+np.arange(10)*0.1,height = bins[:,1], width = widths[:,1],yerr = bins_std[:,1])
        plt.plot(0.05+np.arange(10)*0.1,[0.1 for i in range(10)],'-r')
        plt.plot(0.05+np.arange(10)*0.1,[-0.1 for i in range(10)],'-r')
        plt.title(split)
    plt.savefig(save_path)
    plt.show()

def plot_variable_influence(X_train,X,y,splits):

    #per variable 
    for i in range(len(X[0][0])):
        
        save_path = os.path.join('storage','Results','Setup 5','Variable {}'.format(i))
        plt.title('Variable {} vs dice'.format(i))

        # Plot the models predictions based on single features 
        X_train = [[X[0][k][i]] for k in range(len(X_train))]
        y_train = y[0]

        ridge = Ridge(normalize=False)
        svr = SVR()
        mlp = MLPRegressor((50,100,100,50))

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        ridge.fit(X_train_s,y_train)
        svr.fit(X_train_s,y_train)
        mlp.fit(X_train_s,y_train)

        x_min,x_max = np.min(X_train), np.max(X_train)
        x_span = x_max - x_min
        # take input range to plot model and scale it 
        points = np.reshape(np.arange(x_min,x_max,step=x_span/100),(-1,1))
        points_s = scaler.transform(points)

        y_ridge = ridge.predict(points_s)
        y_svr = svr.predict(points_s)
        y_mlp = mlp.predict(points_s)
               
        plt.plot(points,y_ridge, label='ridge')
        plt.plot(points,y_svr, label='svr')
        plt.plot(points,y_mlp, label='mlp')

        #plot the data points
        for j,split in enumerate(splits):
            filt_X = [[X[j][k][i]] for k in range(len(X[j]))]
            plt.scatter(filt_X,y[j],label=split)

        plt.legend(loc='lower right')
        plt.savefig(save_path)
        plt.show()

def get_l1_losses(truth,pred):    
    return np.absolute(truth-pred)    
    
def main(preprocessing=True,train_density=True,feature_extraction=True,
            extract_dice_scores=True,model_train=True,label=1):
    if preprocessing:
        basic_preprocessing(label)
    if train_density:
        intensities = sample_filtered_intensities(filter_segmentations_dens)
        density_model = Density_model(verbose=False,label=label)
        density_model.train_density(intensities)
        density_model.plot_density()
    if feature_extraction:
        extract_features_all_data(label)
    if extract_dice_scores:
        compute_all_prediction_dice_scores()
    if model_train:
        
        # per model a sublist ridge, svr, mlp
        stds_of_splits = [[],[],[]]   
        all_errors=[[],[],[]]

        scaler = StandardScaler()
        splits = ['train','gc_gc','gc_frankfurt','gc_mosmed','gc_radiopedia']
        X,y = extract_features_train_id_od(filter_feature_extr,splits)

        X_train = scaler.fit_transform(X[0])
        y_train = y[0]
        #plot_variable_influence(X_train,X,y,splits)

        ridge = Ridge(normalize=False)
        svr = SVR()
        mlp = MLPRegressor((50,100,100,50))

        ridge.fit(X_train,y_train)
        svr.fit(X_train,y_train)
        mlp.fit(X_train,y_train)

        for i,split in enumerate(splits):
            
            X_eval = scaler.transform(X[i])
            y_eval = y[i]

            y_ridge = ridge.predict(X_eval)
            y_svr = svr.predict(X_eval)
            y_mlp = mlp.predict(X_eval)

            ridge_err,ridge_std = l1_loss(y_ridge,y_eval, std = True)
            svr_err, svr_std  = l1_loss(y_svr,y_eval,std = True)
            mlp_err, mlp_std = l1_loss(y_mlp,y_eval,std = True)

            l1_loss_bins(y_svr,y_eval,split)

            for i,std in enumerate([ridge_std,svr_std,mlp_std]):
                stds_of_splits[i].append(std)
            for i,errors in enumerate([get_l1_losses(y_ridge,y_eval),get_l1_losses(y_svr,y_eval),get_l1_losses(y_mlp,y_eval)]):
                for err in errors:  
                    all_errors[i].append(err)

            #a vector that predicts the mean value of y_train, is a baseline
            # y_mean = np.mean(y_train)*np.ones(np.shape(y_eval))
            print('{}    : ridge   svr      mlp  '.format(split))
            print('error   {:.3f},   {:.3f},  {:.3f}'.format(ridge_err,svr_err,mlp_err))
            print('std     {:.3f},   {:.3f},  {:.3f}'.format(ridge_std,svr_std,mlp_std))
            # print('Using mean of train values, has error of {} and std of {}'.format(l1_loss(y_mean,y_train,std=True)[0],l1_loss(y_mean,y_train,std=True)[1]))
            print()

        print('TOTAL:                ridge   svr      mlp  ')
        print('mean of split stds   {:.3f},   {:.3f},  {:.3f}'.format(np.mean(stds_of_splits[0]),np.mean(stds_of_splits[1]),np.mean(stds_of_splits[2])))
        print('std of all errors    {:.3f},   {:.3f},  {:.3f}'.format(np.std(all_errors[0]),np.std(all_errors[1]),np.std(all_errors[2])))
        
if __name__ == "__main__":
    main(preprocessing=False,train_density=False,feature_extraction=False,extract_dice_scores=False,model_train=True)
