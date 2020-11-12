# ------------------------------------------------------------------------------
# Hippocampus segmentation task for the HarP dataset
# (http://www.hippocampal-protocol.net/SOPs/index.php)
# ------------------------------------------------------------------------------

import os
import re

import SimpleITK as sitk
import nibabel as nib
import numpy as np

import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
from mp.utils.load_restore import join_path


class VESSEL12(SegmentationDataset):
    r"""Class for the segmentation of the VESSEL12 dataset
    as provided on kaggle https://www.kaggle.com/andrewmvd/lung-vessel-segmentation
    """

    def __init__(self, hold_out_ixs=None):
        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = "VESSEL12"
        name = du.get_dataset_name(global_name)
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Copy the images if not done already
        if not os.path.isdir(dataset_path):
            _extract_images(original_data_path, dataset_path)

        # Fetch all patient/study names
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name in os.listdir(dataset_path))

        # Build instances
        instances = []
        for study_name in study_names:
            instances.append(SegmentationInstance(
                x_path=os.path.join(dataset_path, study_name + '.nii.gz'),
                y_path=os.path.join(dataset_path, study_name + '_gt.nii.gz'),
                name=study_name,
                group_id=None
            ))

        label_names = ['background', 'lung']

        super().__init__(instances, name=name, label_names=label_names,
                         modality='CT', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """
    images_path = os.path.join(source_path, 'ct_scans')
    labels_path = os.path.join(source_path, 'masks')

    filenames = [x for x in os.listdir(images_path) if x.endswith(".mhd")]

    # Create directories
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for filename in filenames:
        # No specific processing
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = sitk.GetArrayFromImage(x)
        y = sitk.ReadImage(os.path.join(labels_path, filename))
        y = sitk.GetArrayFromImage(y)

        # Shape expected: (355, 512, 512)
        # Scans are top down on first axis
        x = np.rot90(x, axes=(1, 0))
        y = np.rot90(y, axes=(1, 0))
        # Scans are now front facing
        assert x.shape == y.shape

        # Save new images so they can be loaded directly
        study_name = filename.split('.mhd')[0]
        sitk.WriteImage(sitk.GetImageFromArray(x), join_path([target_path, study_name + ".nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(y), join_path([target_path, study_name + "_gt.nii.gz"]))
