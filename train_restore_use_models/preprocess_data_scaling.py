import os 
from mp.utils.preprocess_utility_functions import copy_data_into_preprocess_dir,scale_all_images,bring_all_data_into_right_size,extract_features_all_data,compute_all_prediction_dice_scores

def preprocess_data_scaling():
    '''deletes the old data files in preprocess_dir/output_scaled and replaces 
    them with preprocessed (sclaed intensities and resized images/seg/pred if too large) data 
    from data_dir_input
    '''
    if os.environ["INFERENCE_OR_TRAIN"] == 'inference': 
        print('copying data')
        copy_data_into_preprocess_dir()
        print('scaling intensities')
        scale_all_images()
        print('downsizing images, who are too big')
        bring_all_data_into_right_size()
        print('extracting features')
        extract_features_all_data()
    else:
        print("cant use this function in train time")
        RuntimeError

# def preprocess_data_scaling(mode='JIP'):
#     '''deletes the old data files in preprocess_dir/output_scaled and replaces 
#     them with preprocessed data from data_dir_input
#     Args:
#         mode(str): the mode that gets passed to the iterator.
#                     Has to be 'JIP', is just there because i dont know why
#     '''

#     if os.environ["INFERENCE_OR_TRAIN"] == 'inference': 
#         #get paths 
#         input_path = os.path.join(os.environ["WORKFLOW_DIR"],os.environ["OPERATOR_IN_DIR"])
#         output_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"])

#         #delete old images and features
#         _delete_images_and_labels(output_path)

#         #get the scaled version of the images and copy the segmentations
#         ds_iterator = Dataset_Iterator(input_path, mode=mode)
#         ds_iterator.iterate_images(scale_image_save_it,preprocess_mode=True)

#         #compute the features of the segmentations
#         density = Density_model(add_to_name = os.environ["DENSITY_MODEL_NAME"])
#         feat_extr = Feature_extractor(density,['density_distance','dice_scores','connected_components'])
#         for id in os.listdir(output_path):
#             feat_extr.compute_features_id(id)
#     else:
#         print("cant use this function in train time")
#         RuntimeError

    


