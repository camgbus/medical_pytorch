# ------------------------------------------------------------------------------
# Lung task from the Medical Segmentation Decathlon 
# (http://medicaldecathlon.com/)
# ------------------------------------------------------------------------------

# Necessary imports
import os
import numpy as np
import torch
import SimpleITK as sitk
from mp.utils.load_restore import join_path
from mp.data.datasets.dataset_regression import RegressionDataset, RegressionInstance
from mp.paths import storage_data_path, tmp_storage_data_path
import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_augmentation import augment_data_in_four_intensities as augment_data

class DecathlonLung(RegressionDataset):
    r"""Class for the Lung decathlon challenge, contains only
    CT, found at http://medicaldecathlon.com/.
    """
    def __init__(self, subset=None, hold_out_ixs=[], augmented=False):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths    
        global_name = 'DecathlonLung'
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
            instances.append(RegressionInstance(
                x_path=os.path.join(dataset_path, study_name+'.nii.gz'),
                y_label=torch.tensor([1.]),
                name=study_name,
                group_id=None
                ))
        
        # Create augmented images and add them if it is desired
        if augmented:
            aug_data, labels, names = augment_data(instances, 'DecathlonLungAugmented',
                                                   True, False, tmp_storage_data_path)

            # Add to instances
            for name in names:
                instances.append(RegressionInstance(
                    x_path=os.path.join(tmp_storage_data_path, name+'.nii.gz'),
                    y_label=labels[name],
                    name=name,
                    group_id=None
                    ))

        super().__init__(instances, name=global_name,
            modality='CT', nr_channels=1, hold_out_ixs=[])

def _extract_images(source_path, target_path):
    r"""Extracts MRI images and saves the modified images.
    """
    images_path = os.path.join(source_path, 'imagesTr')
    #labels_path = os.path.join(source_path, 'labelsTr')

    # Filenames have the form 'lung_XXX.nii.gz'
    filenames = [x for x in os.listdir(images_path) if x[:4] == 'lung']

    # Create directories if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for filename in filenames:

        # Extract all images (3D)
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = sitk.GetArrayFromImage(x)

        # Save new images so they can be loaded directly
        study_name = filename.replace('_', '').split('.nii')[0]
        sitk.WriteImage(sitk.GetImageFromArray(x), 
            join_path([target_path, study_name+".nii.gz"]))
