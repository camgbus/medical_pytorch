import numpy as np
import SimpleITK as sitk
import os 
from mp.utils.Iterators import Dataset_Iterator
import shutil

def preprocess_data_scaling(mode='JIP'):
    '''deletes the old data files in preprocess_dir/output_scaled and replaces them with preprocessed data from data_dir_input'''
    input_path =   os.path.join(os.environ["WORKFLOW_DIR"],os.environ["OPERATOR_IN_DIR"])
    output_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"])
    _delete_images_and_labels(output_path)
    ds_iterator = Dataset_Iterator(input_path, mode=mode)
    ds_iterator.iterate_images(scale_image_save_it,preprocess_mode=True)
    
def scale_image_save_it(img_path,seg_path,name,ending):
    '''takes a name to an image, scales the image to [0,1] and then saves it in the appropriate format in 
    preprocessed_dir/output_scaled/id/img/....nii'''
    img = scale_image(img_path)
    save_preprocessed_img_seg(img,seg_path,name,ending)

def save_preprocessed_img_seg(img,seg_path,name,ending):
    '''takes and image and saves it in PREPROCESSED_OPERATOR_OUT_SCALED_DIR'''
    #get the paths 
    save_path_i = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],name,'img')
    save_path_s = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],name,'seg')
    save_path_img = os.path.join(save_path_i,'img.{}'.format(ending))
    save_path_seg = os.path.join(save_path_s,'001.{}'.format(ending))

    #make directories 
    if not os.path.isdir(save_path_i):
        os.makedirs(save_path_i)
    if not os.path.isdir(save_path_s):
        os.makedirs(save_path_s)

    # save the img/seg
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img,save_path_img)
    shutil.copyfile(seg_path,save_path_seg)

def scale_image(img_path,d_type=np.float32):
    r''' takes a path to an image and computes the a version with the values scaled to [0,1]. 
    Args:
        img_path(str): the path to the image
        d_type (np.datatype): a datatype the image shall have as output
    
    Returns (nd.array): the scaled image
    '''
    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)

    max_val = np.max(img)
    min_val = np.min(img)
    span = max_val - min_val
    if span == 0:
        print('The image has only one intensity value and thus cannot be rescaled')
        return RuntimeError
        
    shape = np.shape(img)

    add_array = np.ones(shape)*min_val
    img = img - add_array
    img = img * 1/span  
    img = np.around(img,decimals=4)
    if d_type:
        img = np.array(img,dtype=d_type)
    return img

def _delete_images_and_labels(path):
    r"""This function deletes every nifti and json (labels) file in the path."""
    # Walk through path and delete all .nii files
    print('Walk trough directory \'{}\' and delete nifti files..'.format(path))
    for dname, dirs, files in os.walk(path):
        for num, fname in enumerate(files):
            msg = str(num + 1) + '_ of ' + str(len(files)) + '_ file(s).'
            print (msg, end = '\r')
            # Check if file is a nifti file and delete it
            if '.nii' in fname or '.json' in fname:
                fpath = os.path.dirname(dname)
                shutil.rmtree(fpath)