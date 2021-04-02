import os 
import shutil
from train_restore_use_models.preprocess_data_scaling import scale_image_save_it
gt_data = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"],'Covid-RACOON','All images and labels')
pred_data = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"],'Covid-RACOON','All predictions')

extension = os.environ["INPUT_FILE_ENDING"]

if not os.environ["INFERENCE_OR_TRAIN"] == 'train': 
    print('Cant preprocess train data in mode other then train')
    RuntimeError

overall_nr_pred = 0

def preprocess_data_scaling_train():
    for task in os.listdir(gt_data):
        ids = [id.split('_')[0]  for id in os.listdir(os.path.join(gt_data,task,'imagesTr'))]
        for id in ids:
            img_path = os.path.join(gt_data,task,'imagesTr',id+'_0000.'+extension)
            seg_path = os.path.join(gt_data,task,'labelsTr',id+'.'+extension)
            name = task + '_' + id
            scale_image_save_it(img_path,seg_path,name)
            load_predictions(task,id,name)
    print(overall_nr_pred)
        
def load_predictions(task,id,name):
    nr_pred = 0
    for model in os.listdir(pred_data):
        origin_pred_path = os.path.join(pred_data,model,task,id+'.'+extension)
        if os.path.exists(origin_pred_path):
            pred_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name,'pred')
            dst_pred_path = os.path.join(pred_path,str(nr_pred)+'.nii.gz')
            if not os.path.isdir(pred_path):
                os.makedirs(pred_path)
            shutil.copyfile(origin_pred_path,dst_pred_path)
            nr_pred =+ 1
            overall_nr_pred =+ 1
            
            
