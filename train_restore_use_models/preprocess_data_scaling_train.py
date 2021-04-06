import os 
from mp.utils.preprocess_utility_functions import copy_data_into_preprocess_dir,scale_all_images,bring_all_data_into_right_size,extract_features_all_data,compute_all_prediction_dice_scores

def preprocess_data_scaling_train():
    '''goes through the prespecified train_dir/... and 
        1. deletes the files in the preprocess/.._scaled
        goes through the img-seg folder and id-wise
            2. extracts the scaled images and the segmentations
            3. extracts the predictions if existing 
    '''
    if os.environ["INFERENCE_OR_TRAIN"] == 'train': 
        copy_data_into_preprocess_dir()
        scale_all_images()
        bring_all_data_into_right_size()
        extract_features_all_data()
        compute_all_prediction_dice_scores()
    else: 
        print("cant use this function in inference time")
        RuntimeError

# def preprocess_data_scaling_train():
#     '''goes through the prespecified train_dir/... and 
#         1. deletes the files in the preprocess/.._scaled
#         goes through the img-seg folder and id-wise
#             2. extracts the scaled images and the segmentations
#             3. extracts the predictions if existing 
#     '''
#     if os.environ["INFERENCE_OR_TRAIN"] == 'train': 
        
#         #get the paths 
#         output_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
#         gt_data = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_GT"])
        
#         #delte the old images and features 
#         _delete_images_and_labels(output_path)

#         #load the scaled images, segmentations and predictions with dice scores into the preprocessed directory
#         for task in os.listdir(gt_data):
#             ids = [id.split('_')[0]  for id in os.listdir(os.path.join(gt_data,task,'imagesTr'))]
#             for id in ids:
#                 img_path = os.path.join(gt_data,task,'imagesTr',id+'_0000.'+os.environ["INPUT_FILE_ENDING"])
#                 seg_path = os.path.join(gt_data,task,'labelsTr',id+'.'+os.environ["INPUT_FILE_ENDING"])
#                 name = task + '_' + id
#                 scale_image_save_it(img_path,seg_path,name)
#                 load_predictions_and_dice(task,id,name)
#     else: 
#         print("cant use this function in inference time")
#         RuntimeError





        
            
