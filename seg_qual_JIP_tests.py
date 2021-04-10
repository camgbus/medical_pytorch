import os 
from mp.paths import JIP_dir
#from train_restore_use_models.train_retrain_density import train_density
#from train_restore_use_models.train_retrain_dice_predictor import train_dice_predictor
#from mp.utils.intensities import get_intensities
# from mp.utils.feature_extractor import Feature_extractor
#from mp.models.densities.density import Density_model
# from mp.models.regression.dice_predictor import Dice_predictor
import numpy as np
# from mp.quantifiers.IntBasedQuantifier import IntBasedQuantifier
# from mp.data.DataConnectorJIP import DataConnector
# from train_restore_use_models.preprocess_data_scaling import preprocess_data_scaling
# from train_restore_use_models.preprocess_data_scaling_train import preprocess_data_scaling_train

#set environmental variables
os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
os.environ["OPERATOR_IN_DIR"] = "input_small"
os.environ["OPERATOR_OUT_DIR"] = "output"
os.environ["OPERATOR_TEMP_DIR"] = "temp"
os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')

os.environ["PREPROCESSED_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'preprocessed_dirs')
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"] = "output_scaled"
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = "output_scaled_train"

os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')
os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('Covid-RACOON','All images and labels')
os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('Covid-RACOON','All predictions')

# os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('gt_small')
# os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('pred_small')

os.environ["INFERENCE_OR_TRAIN"] = 'train'

os.environ["INPUT_FILE_ENDING"] = 'nii.gz'

os.environ["DENSITY_MODEL_NAME"] = 'dummy'

#The following tests wokr per se, but are not fitted to the structure of incoming new train data
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
    save_name='delete_me'
    save_descr='please delete me '
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
    y_train = np.ones(30)
    data_describtion = 'using all of the data of UK_frankfurt with random labels. Features are con_comp, density_dist, dice_scores'
    model_describtion = 'a MLP model, further specs, see below'
    verbose = True 

    train_dice_predictor(model_name=model_name,feature_names=feature_names,dens_model=dens_model,dens_add_name=dens_add_name,
                            list_of_paths=list_of_paths, names_extracted_features=names_extracted_features ,y_train=y_train ,
                            data_describtion = data_describtion, model_describtion = model_describtion ,verbose=verbose , 
                            solver ='adam',learning_rate='adaptive',hidden_layer_sizes=(10,30,50,50,20))

    print('Everything went through, so should be fine')

def test_int_quantifier_working():
    img_path = os.path.join(JIP_dir, 'data_dirs','input_small','FRACorona_KGU-8160FACFB08D','img','img.nii.gz')
    seg_path = os.path.join(JIP_dir, 'data_dirs','input_small','FRACorona_KGU-8160FACFB08D','seg','001.nii.gz')
    img_instance = [img_path,img_path]
    seg_instance = [seg_path,seg_path]
    quantifier = IntBasedQuantifier()
    print(quantifier.get_quality(seg_instance,img_instance))



#The following tests are working on the JIP structure and use environ vars for that purpose

### Inference Workflow 
def test_inference_preprocess_workflow():
    from mp.utils.preprocess_utility_functions import basic_preprocessing
    basic_preprocessing()

### Train Workflow
def test_train_workflow(preprocess=True,train_density=True,train_dice_pred=True,verbose=True):
    from train_restore_use_models.train_int_based_quantifier import train_int_based_quantifier
    train_int_based_quantifier(preprocess,train_density,train_dice_pred,verbose)

#working even though dice predictor performs terrible
#test_train_workflow(preprocess=False,train_density=False)

from mp.utils.feature_extractor import Feature_extractor 
from sklearn.model_selection import train_test_split

feat_extr = Feature_extractor()
X,y = feat_extr.collect_train_data()

def scale_arr(arr):

    max_val = np.max(arr)
    min_val = np.min(arr)
    span = max_val - min_val

    shape = np.shape(arr)
    add_array = np.ones(shape)*min_val
    arr = arr - add_array
    arr = arr * 1/span  


