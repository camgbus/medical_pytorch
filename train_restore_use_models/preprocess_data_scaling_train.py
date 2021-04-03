import os 
import shutil
import SimpleITK as sitk
import json
from train_restore_use_models.preprocess_data_scaling import scale_image_save_it,_delete_images_and_labels
from mp.eval.metrics.simple_scores import dice_score
from mp.utils.feature_extractor import Feature_extractor
from mp.models.densities.density import Density_model

def preprocess_data_scaling_train():

    if os.environ["INFERENCE_OR_TRAIN"] == 'train': 
        
        #get the paths 
        output_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
        gt_data = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_GT"])
        
        #delte the old images and features 
        _delete_images_and_labels(output_path)

        #load the scaled images, segmentations and predictions with dice scores into the preprocessed directory
        for task in os.listdir(gt_data):
            ids = [id.split('_')[0]  for id in os.listdir(os.path.join(gt_data,task,'imagesTr'))]
            for id in ids:
                img_path = os.path.join(gt_data,task,'imagesTr',id+'_0000.'+os.environ["INPUT_FILE_ENDING"])
                seg_path = os.path.join(gt_data,task,'labelsTr',id+'.'+os.environ["INPUT_FILE_ENDING"])
                name = task + '_' + id
                scale_image_save_it(img_path,seg_path,name)
                load_predictions_and_dice(task,id,name)

        #compute the features for the img-seg and img-pred pairs
        get_features_of_prepr_data()
    else: 
        print("cant use this function in inference time")
        RuntimeError

def get_features_of_prepr_data():
    density = Density_model(add_to_name = os.environ["DENSITY_MODEL_NAME"])
    feat_extr = Feature_extractor(density,['density_distance','dice_scores','connected_components'])
    for id in os.listdir(os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])):
        feat_extr.compute_features_id(id)

def load_predictions_and_dice(task,id,name):
    nr_pred = 0
    
    pred_data = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_PRED"])
    #iterate over all models, that made predictions 
    for model in os.listdir(pred_data):
        
        #look up, if there is a prediction for the img-seg pair
        origin_pred_path = os.path.join(pred_data,model,task,id+'.'+os.environ["INPUT_FILE_ENDING"])
        if os.path.exists(origin_pred_path):
            path_to_id = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name)
            pred_path = os.path.join(path_to_id,'pred','pred_{}'.format(nr_pred))

            #copy the prediction
            dst_pred_path = os.path.join(pred_path,'pred_'+str(nr_pred)+'.nii.gz')
            if not os.path.isdir(pred_path):
                os.makedirs(pred_path)
            shutil.copyfile(origin_pred_path,dst_pred_path)

            #compute and save the dice score
            dice_score_save_path = os.path.join(pred_path,'dice_score.json')
            target_path = os.path.join(path_to_id,'seg','001.nii.gz')
            target = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(dst_pred_path))
            dice = dice_score(target,prediction)
            with open(dice_score_save_path,'w') as file:
                json.dump(dice,file)

            nr_pred =+ 1

        
            
