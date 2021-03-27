import os 
from mp.paths import JIP_dir
from train_restore_use_models.train_retrain_density import train_density
from mp.utils.intensities import get_intensities
from mp.utils.feature_extractor import Feature_extractor
from mp.models.densities.density import Density_model
from mp.models.regression
import numpy as np

#set environmental variables
os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
os.environ["OPERATOR_IN_DIR"] = "input"
os.environ["OPERATOR_OUT_DIR"] = "output"
os.environ["OPERATOR_TEMP_DIR"] = "temp"
os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')

def test_train_density_working():
    
    #set the params 
    model='gaussian_kernel'
    ending_of_model='UK_Fra'
    list_of_paths=[]#[os.path.join(os.environ['WORKFLOW_DIR'],os.environ['OPERATOR_IN_DIR'])]
    data_mode = 'JIP_test'
    data_describtion = 'all 30 img-seg pairs from UK Frankfurt, new pairs, new data format'
    model_describtion = 'gaussian_kernel with bw 0.005'
    precom_intensities = ['UK_Fra']# ['dummy_int']
    verbose = False 
    #test, if density model works
    train_density(model,ending_of_model,list_of_paths,data_mode,
                    data_describtion,model_describtion,precom_intensities,verbose, bandwidth=0.005)

    print('Everything went through, so should be fine')

def test_get_intensities_working():
    mode = 'JIP_test'
    save = True 
    save_name = 'UK_Fra'
    save_descr = 'Intensities of all 30 img-seg pairs in UK_Fra'
    list_of_paths=[os.path.join(os.environ['WORKFLOW_DIR'],os.environ['OPERATOR_IN_DIR'])]
    verbose = True
    get_intensities(list_of_paths, min_size=100, mode=mode,save = save, save_name=save_name, save_descr=save_descr, verbose=True)
    print('Everything went through, so should be fine')

def test_feature_extractor_working():
    list_of_paths=[os.path.join(os.environ['WORKFLOW_DIR'],os.environ['OPERATOR_IN_DIR'])]
    save = True
    save_name='UK_Fra_density_distances'
    save_descr='all 30 img-seg pairs from UK Fra. Computed averaged density distances'
    mode = 'JIP_test'
    features = ['density_distance','dice_scores','connected_components']

    density = Density_model(add_to_name='UK_Fra')
    
    feat_extr = Feature_extractor(density=density,features=features)
    feat_arr = feat_extr.get_features_from_paths(list_of_paths,mode=mode,save=save,
                save_name=save_name,save_descr=save_descr)
    print(feat_arr)
    print('Everything went through, so should be fine')

def test_dice_predictor_working():
    model_name = 'UK_Fra_dummy'
    feature_names = ['density_distance','dice_scores','connected_components']
    dens_model = 'gaussian'
    dens_add_name = 'UK_Fra'
    list_of_paths = []
    names_extracted_features = ['UK_Fra']
    #get a random vector of 30 labels, since no dice scores are present
    y_train = np.random.rand(30)
    data_describtion = 'using all of the data of UK_frankfurt with random labels'
    model_describtion = 'a MLP model, further specs, see below'
    verbose = True 

    train_dice_predictor(model_name=model_name,feature_names=feature_names,dens_model=dens_model,dens_add_name=dens_add_name,
                            list_of_paths=list_of_paths, names_extracted_features=names_extracted_features ,y_train=y_train ,
                            data_describtion = data_describtion, model_describtion = model_describtion ,verbose=verbose , 
                            solver ='adam',lr='adaptive',hidden_layer_sizes=(10,30,50,50,20))

#test_get_intensities_working()
#test_train_density_working()
#test_feature_extractor_working()



    