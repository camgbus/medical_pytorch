# Import needed libraries
from mp.data.datasets.dataset_JIP_cnn import JIPDataset
import numpy as np
import SimpleITK as sitk
def preprocess_data(config):
    r"""This function is used to load the original data from the workflow and preprocesses it
        by saving it in the preprocessed workflow."""
    
    JIP = JIPDataset(img_size=config['input_shape'], max_likert_value=config['max_likert_value'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot=config['msg_bot'],\
                     nr_images=config['nr_images'])
    return JIP.preprocess()

def preprocess_data_intensities(img_path,d_type=np.float32):
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