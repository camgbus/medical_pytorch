# ------------------------------------------------------------------------------
# Left Atrium segmentation task from the Medical Segmentation Decathlon 
# (http://medicaldecathlon.com/)
# ------------------------------------------------------------------------------

# Necessary imports
import os
import numpy as np
import SimpleITK as sitk
from mp.utils.load_restore import join_path
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
import mp.data.datasets.dataset_utils as du

class DecathlonLeftAtrium(SegmentationDataset):
    r"""Class for the Left Atrium segmentation decathlon challenge, contains only
    MRI, found at http://medicaldecathlon.com/.
    """
    def __init__(self, subset=None, hold_out_ixs=[]):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths    
        global_name = 'DecathlonLeftAtrium'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Extract all images, if not already done
        if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
            _extract_images(original_data_path, dataset_path)

        # Fetch all patient/study names that do not begin with '._'
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
            in os.listdir(dataset_path) if '._' not in file_name)

        # Build instances
        instances = []
        for study_name in study_names:
            instances.append(SegmentationInstance(
                x_path=os.path.join(dataset_path, study_name+'.nii.gz'),
                y_path=os.path.join(dataset_path, study_name+'_gt.nii.gz'),
                name=study_name,
                group_id=None
                ))
                
        label_names = ['background', 'left atrium']
        
        super().__init__(instances, name=global_name, label_names=label_names, 
            modality='MRI', nr_channels=1, hold_out_ixs=[])

def _extract_images(source_path, target_path):#, merge_labels):
    r"""Extracts MRI images and saves the modified images.
    """
    images_path = os.path.join(source_path, 'imagesTr')
    labels_path = os.path.join(source_path, 'labelsTr')

    # Filenames have the form 'la_XXX.nii.gz'
    filenames = [x for x in os.listdir(images_path) if x[:2] == 'la']

    # Create directories if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for filename in filenames:

        # Extract all images (3D)
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = sitk.GetArrayFromImage(x)
        y = sitk.ReadImage(os.path.join(labels_path, filename))
        y = sitk.GetArrayFromImage(y)
        assert x.shape == y.shape

        # Save new images so they can be loaded directly
        study_name = filename.replace('_', '').split('.nii')[0]
        sitk.WriteImage(sitk.GetImageFromArray(x), 
            join_path([target_path, study_name+".nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(y), 
            join_path([target_path, study_name+"_gt.nii.gz"]))
