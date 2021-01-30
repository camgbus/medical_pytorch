import os
import re

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch

import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
from mp.utils.load_restore import join_path
import mp.data.pytorch.transformation as trans


class UKF2(SegmentationDataset):
    r"""class for covid data from UK_Frankfurt2
    """

    def __init__(self, hold_out_ixs=None):
        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = "UKF2"
        name = du.get_dataset_name(global_name)
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Copy the images if not done already
        if not os.path.isdir(dataset_path):
            _extract_images(original_data_path, dataset_path)

        # Fetch all patient/study names
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
            in os.listdir(dataset_path))

        # Build instances
        instances = []
        for study_name in study_names:
            instances.append(SegmentationInstance(
                x_path=os.path.join(dataset_path, study_name + '.nii.gz'),
                y_path=os.path.join(dataset_path, study_name + '_gt.nii.gz'),
                name=study_name,
                group_id=None
            ))

        label_names = ['background', 'consolidation']

        super().__init__(instances, name=name, label_names=label_names,
                         modality='CT', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    # Create directories
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    
    filenames = [x for x in os.listdir(source_path)]

    for filename in filenames:
        image_path = os.path.join(source_path,filename,'image.nii.gz')
        label_path = os.path.join(source_path,filename,'mask.nii.gz')
   
        # No specific processing
        x = sitk.ReadImage(image_path)
        x = sitk.GetArrayFromImage(x)
        shape = np.size(x)
        x = torch.from_numpy(x)
        x = torch.unsqueeze(x,0)
        y = sitk.ReadImage(label_path)
        y = sitk.GetArrayFromImage(y)
        y = torch.from_numpy(y)
        y = torch.unsqueeze(y,0)
        img_size = (1,shape[0],256,256)

        try:
            x = trans.resize_3d(x, img_size)
        except:
            print('Image could not be resized and will therefore be skipped: {}.'
            .format(filename))
            continue
        x = torch.squeeze(x)
        
        try:
            y = trans.resize_3d(y, img_size, label=True)
        except:
            print('Image could not be resized and will therefore be skipped: {}.'
            .format(filename))
            continue
        y = torch.squeeze(y)


        # Save new images so they can be loaded directly
        study_name = filename
        sitk.WriteImage(sitk.GetImageFromArray(x), join_path([target_path, study_name + ".nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(y), join_path([target_path, study_name + "_gt.nii.gz"]))