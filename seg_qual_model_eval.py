import os 
from mp.paths import JIP_dir
import numpy as np
import torchio
import torch
from skimage.measure import label,regionprops
import matplotlib.pyplot as plt
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
            seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
            img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
            labeled_image, nr_components = label(seg, return_num=True)
            props = regionprops(labeled_image)
            props = sorted(props ,reverse=True, key =lambda dict:dict['area'])
            nr_components = len(props)
            comp = 0
            while comp < nr_components and props[comp].area > 100:
                output.append(sample_intensities(img,seg,props[comp],number=750))
                comp += 1 
    return output

def filter_feature_extr(id,model):
    if model[:7] in ['Task740'] and id[:7] in ['Task740','Task541']:
        return 'train'
    if model[:7] in ['Task740'] and id[:7] in ['Task542','Task741']:
        return 'id'
    if model[:7] in ['Task740'] and id[:7] not in ['Task740','Task541','Task741','Task542']:
        return 'od'
    return 'other'

def extract_features_train_id_od(filter):
    X_train = []
    X_id= []
    X_od = []
    y_train = []
    y_id = []
    y_od = []
    dens = Density_model()
    feat_extr = Feature_extractor(dens)
    work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    for id in os.listdir(work_path):
        all_pred_path = os.path.join(work_path,id,'pred')
        if os.path.exists(all_pred_path):
            for model in os.listdir(all_pred_path):
                split = filter(id,model)
                if split in ['train','id','od']:
                    feature_path = os.path.join(all_pred_path,model,'features.json')
                    label_path = os.path.join(all_pred_path,model,'dice_score.json')
                    feat_vec = feat_extr.read_feature_vector(feature_path)
                    #feat_vec = [feat_vec[0]]
                    label = feat_extr.read_prediction_label(label_path)
                    if np.isnan(np.sum(np.array(feat_vec))) or feat_vec[0]>100000:
                        pass 
                    else:
                        if filter(id,model)=='train':
                            X_train.append(feat_vec)
                            y_train.append(label)
                        if filter(id,model)=='id':
                            X_id.append(feat_vec)
                            y_id.append(label)
                        if filter(id,model)=='od':
                            X_od.append(feat_vec)
                            y_od.append(label)
    return X_train, X_id, X_od, y_train, y_id, y_od

def l2_loss(pred,truth):
    n = len(pred)
    return (1/n)*(np.sum((pred - truth)**2))

def l1_loss(pred,truth):
    n=len(pred)
    return (1/n)*np.sum(np.absolute(pred-truth))

def main(preprocessing=True,train_density=True,feature_extraction=True,extract_dice_scores=True,model_train=True,label=1):
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
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor

        scaler = StandardScaler()
        X_train, X_id, X_od, y_train, y_id, y_od = extract_features_train_id_od(filter_feature_extr)
        print(len(X_train), len(X_id), len(X_od),len(y_train), len(y_id), len(y_od))

        X_train_scaled = scaler.fit_transform(X_train)

        ridge = Ridge(normalize=False)
        svr = SVR()
        mlp = MLPRegressor((50,100,100,50))

        ridge.fit(X_train_scaled,y_train)
        svr.fit(X_train_scaled,y_train)
        mlp.fit(X_train_scaled,y_train)

        y_ridge_train = ridge.predict(X_train_scaled)
        y_svr_train = svr.predict(X_train_scaled)
        y_mlp_train = mlp.predict(X_train_scaled)

        # #evaluation of density
        # X = np.append(X_train,X_id)
        # X = np.append(X,X_od)

        # y = np.append(y_train,y_id)
        # y = np.append(y,y_od)

        # plt.scatter(X_train,y_train)
        # plt.scatter(X_id,y_id, label='id') 
        # plt.scatter(X_od,y_od, label='od')       
        # plt.scatter(X_train,y_ridge_train, label='ridge')
        # plt.scatter(X_train,y_svr_train, label='svr')
        # plt.scatter(X_train,y_mlp_train, label='mlp')
        # plt.legend(loc='lower left')
        # plt.show()
        # return 

        ridge_train_loss = l2_loss(y_ridge_train,y_train)
        svr_train_loss = l2_loss(y_svr_train,y_train)
        mlp_train_loss = l2_loss(y_mlp_train,y_train)

        ridge_train_err = l1_loss(y_ridge_train,y_train)
        svr_train_err = l1_loss(y_svr_train,y_train)
        mlp_train_err = l1_loss(y_mlp_train,y_train)

        print('TRAINING: ridge   svr      mlp  ')
        print('error   {:.3f},   {:.3f},  {:.3f}'.format(ridge_train_err,svr_train_err,mlp_train_err))
        print('loss    {:.3f},   {:.3f},  {:.3f}'.format(ridge_train_loss,svr_train_loss,mlp_train_loss))
        print()

        if X_id: 
            X_id_scaled = scaler.transform(X_id)

            y_ridge_id = ridge.predict(X_id_scaled)
            y_svr_id = svr.predict(X_id_scaled)
            y_mlp_id = mlp.predict(X_id_scaled)

            ridge_id_loss = l2_loss(y_ridge_id,y_id)
            svr_id_loss = l2_loss(y_svr_id,y_id)
            mlp_id_loss = l2_loss(y_mlp_id,y_id)

            ridge_id_err = l1_loss(y_ridge_id,y_id)
            svr_id_err = l1_loss(y_svr_id,y_id)
            mlp_id_err = l1_loss(y_mlp_id,y_id)
        
            print('In Distr: ridge   svr      mlp  ')
            print('error     {:.3f}, {:.3f}, {:.3f}'.format(ridge_id_err,svr_id_err,mlp_id_err))
            print('loss     {:.3f},  {:.3f}, {:.3f}'.format(ridge_id_loss,svr_id_loss,mlp_id_loss))
            print()

        if X_od:
            X_od_scaled = scaler.transform(X_od)

            y_ridge_od = ridge.predict(X_od_scaled)
            y_svr_od = svr.predict(X_od_scaled)
            y_mlp_od = mlp.predict(X_od_scaled)
        
            ridge_od_loss = l2_loss(y_ridge_od,y_od)
            svr_od_loss = l2_loss(y_svr_od,y_od)
            mlp_od_loss = l2_loss(y_mlp_od,y_od)

            ridge_od_err = l1_loss(y_ridge_od,y_od)
            svr_od_err = l1_loss(y_svr_od,y_od)
            mlp_od_err =  l1_loss(y_mlp_od,y_od)

            print('Out Distr: ridge   svr       mlp  ')
            print('error    {:.3f},   {:.3f},  {:.3f}'.format(ridge_od_err,svr_od_err,mlp_od_err))
            print('loss     {:.3f},   {:.3f},  {:.3f}'.format(ridge_od_loss,svr_od_loss,mlp_od_loss))


if __name__ == "__main__":
    main(preprocessing=False,train_density=False,feature_extraction=False,extract_dice_scores=False,model_train=True)

def test_models_accuracy(times=50, mode=1):
    from mp.utils.feature_extractor import Feature_extractor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from mp.models.densities.density import Density_model
    import matplotlib.pyplot as plt 

    dens = Density_model(label=1)
    feature_extractor = Feature_extractor(dens)

    if mode == 0:
        X,y = feature_extractor.collect_train_data()
        # plt.hist(y,20,(0,1))
        # plt.show()

        losses_ridge = []
        losses_svr = []
        losses_mlp = []
        for test_size in [0.3]:
            for _ in range(times):

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                ridge = Ridge(normalize=False)
                svr = SVR()
                mlp = MLPRegressor((50,100,100,50))

                ridge.fit(X_train,y_train)
                svr.fit(X_train,y_train)
                mlp.fit(X_train,y_train)

                y_ridge = ridge.predict(X_test)
                y_svr = svr.predict(X_test)
                y_mlp = mlp.predict(X_test)

                losses_ridge.append(l2_loss(y_test,y_ridge))
                losses_svr.append(l2_loss(y_test,y_svr))
                losses_mlp.append(l2_loss(y_test,y_mlp))

            print(test_size)
            print(np.max(y_svr),np.min(y_svr))
            print(np.max(y_test),np.min(y_test))
            print((1/len(y_test))*np.sum(np.absolute(y_svr-y_test)))
            print()
            print(losses_svr[:10])
            print('ridge',np.mean(losses_ridge),np.var(losses_ridge))
            print()
            print('svr',np.mean(losses_svr),np.var(losses_svr))
            print()
            print('mlp',np.mean(losses_mlp),np.var(losses_mlp))

    if mode == 1:
        X_train,X_test,y_train,y_test = feature_extractor.collect_train_data_split()
        # plt.hist(y_train,10,(0,1))
        # plt.show()
        # plt.hist(y_test,10,(0,1))
        # plt.show()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        ridge = Ridge(normalize=False)
        svr = SVR()
        mlp = MLPRegressor((50,100,100,50))

        ridge.fit(X_train,y_train)
        svr.fit(X_train,y_train)
        mlp.fit(X_train,y_train)

        y_ridge = ridge.predict(X_test)
        print(np.absolute(y_ridge-y_test))
        y_svr = svr.predict(X_test)
        y_mlp = mlp.predict(X_test)

        print('ridge loss :{} , svr loss: {}, mlp loss: {}'.format(l2_loss(y_test,y_ridge),l2_loss(y_test,y_svr),l2_loss(y_test,y_mlp)))
        print('ridge acc :{} , svr acc: {}, mlp acc: {}'.format(l1_loss(y_test,y_ridge),l1_loss(y_test,y_svr),l1_loss(y_test,y_mlp)))