import os 
from mp.paths import JIP_dir
from train_restore_use_models.train_retrain_density import train_density
from mp.utils.intensities import get_intensities
from mp.utils.feature_extractor import Feature_extractor
from mp.models.densities.density import Density_model
#set environmental variables
os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
os.environ["OPERATOR_IN_DIR"] = "input_small"
os.environ["OPERATOR_OUT_DIR"] = "output"
os.environ["OPERATOR_TEMP_DIR"] = "temp"
os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')

def test_train_density_working():
    
    #set the params 
    model='gaussian_kernel'
    ending_of_model='dummy'
    list_of_paths=[os.path.join(os.environ['WORKFLOW_DIR'],os.environ['OPERATOR_IN_DIR'])]
    data_mode = 'normal'
    data_describtion = '3 img-seg pairs from UK Frankfurt'
    model_describtion = 'gaussian_kernel with bw 0.005'
    precom_intensities = []# ['dummy_int']
    verbose = True 
    #test, if density model works
    train_density(model,ending_of_model,list_of_paths,data_mode,
                    data_describtion,model_describtion,precom_intensities,verbose, bandwidth=0.005)

    print('Everything went through, so should be fine')

def test_get_intensities_working():
    list_of_paths=[os.path.join(os.environ['WORKFLOW_DIR'],os.environ['OPERATOR_IN_DIR'])]
    get_intensities(list_of_paths, min_size=100, mode='normal',save = False, save_name=None, save_descr=None)
    print('Everything went through, so should be fine')

def test_feature_extractor_working():
    list_of_paths=[os.path.join(os.environ['WORKFLOW_DIR'],os.environ['OPERATOR_IN_DIR'])]
    save = False
    save_name='dummy_features'
    save_descr='3 img-seg pairs from UK Fra'
    mode = 'normal'
    features = ['density_distance'] #['density_distance','dice_scores','connected_components']

    density = Density_model(add_to_name='dummy')
    
    feat_extr = Feature_extractor(density=density,features=features)
    feat_arr = feat_extr.get_features_from_paths(list_of_paths,mode=mode,save=save,
                save_name=save_name,save_descr=save_descr)
    print(feat_arr)
    print('Everything went through, so should be fine')

test_feature_extractor_working()
    


    