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
from mp.utils.mask_bounding_box import mask_bbox_3D
from mp.utils.load_restore import join_path


class HarP(SegmentationDataset):
    r"""Class for the segmentation of the HarP dataset,
    found at http://www.hippocampal-protocol.net/SOPs/index.php
    with the masks as .nii files and the scans as .mnc files.
    """

    def __init__(self, subset=None, hold_out_ixs=None):
        # Part is either: "Training", "Validation" or "All"
        default = {"Part": "All"}
        if subset is not None:
            default.update(subset)
            subset = default
        else:
            subset = default

        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = 'HarP'
        name = du.get_dataset_name(global_name, subset)
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Build instances
        instances = []
        folders = []
        if subset["Part"] in ["Training", "All"]:
            folders.append(("100", "Training"))
        if subset["Part"] in ["Validation", "All"]:
            folders.append(("35", "Validation"))

        for orig_folder, dst_folder in folders:
            # Paths with the sub-folder for the current subset
            dst_folder_path = os.path.join(dataset_path, dst_folder)

            # Copy the images if not done already
            if not os.path.isdir(dst_folder_path):
                _extract_images(original_data_path, dst_folder_path, orig_folder)

            # Fetch all patient/study names
            study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name
                              in os.listdir(os.path.join(dataset_path, dst_folder)))

            for study_name in study_names:
                instances.append(SegmentationInstance(
                    x_path=os.path.join(dataset_path, dst_folder, study_name + '.nii.gz'),
                    y_path=os.path.join(dataset_path, dst_folder, study_name + '_gt.nii.gz'),
                    name=study_name,
                    group_id=None
                ))

        label_names = ['background', 'hippocampus']

        super().__init__(instances, name=name, label_names=label_names,
                         modality='T1w MRI', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path, subset):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    # Folder 100 is for training (100 subjects), 35 subjects are left over for validation
    affine = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    images_path = os.path.join(source_path, subset)
    labels_path = os.path.join(source_path, f'Labels_{subset}_NIFTI')

    # Create directories
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    files_with_swapped_masks = {"ADNI_007_S_1304_74384_ACPC.mnc",
                                "ADNI_016_S_4121_280306_ACPC.mnc",
                                "ADNI_029_S_4279_265980_ACPC.mnc",
                                "ADNI_136_S_0429_109839_ACPC.mnc"}

    # For each MRI, there are 2 segmentation (left and right hippocampus)
    for filename in os.listdir(images_path):
        # Loading the .mnc file and converting it to a .nii.gz file
        minc = nib.load(os.path.join(images_path, filename))
        x: np.array = nib.Nifti1Image(np.asarray(minc.dataobj), affine=affine).get_data()

        # We need to recover the study name of the image name to construct the name of the segmentation files
        match = re.match(r"ADNI_[0-9]+_S_[0-9]+_[0-9]+", filename)
        if match is None:
            raise Exception(f"A file ({filename}) does not match the expected file naming format")

        # For each side of the brain
        for side in ("_L", "_R"):
            study_name = match[0] + side

            y = sitk.ReadImage(os.path.join(labels_path, study_name + ".nii"))
            y = sitk.GetArrayFromImage(y)

            # Shape expected: (189, 233, 197)
            assert x.shape == y.shape
            # BUGFIX: Some segmentation have some weird values eg {26896.988, 26897.988} instead of {0, 1}
            y = (y - np.min(y.flat)).astype(np.uint32)

            # Cropping bounds computed to fit the ground truth
            if (side == "_L") ^ (filename in files_with_swapped_masks):
                y = y[40: 104, 78: 142, 49: 97]
                x_cropped = x[40: 104, 78: 142, 49: 97]
            else:
                y = y[40: 104, 78: 142, 97: 145]
                x_cropped = x[40: 104, 78: 142, 97: 145]

            # Need to do move an axis as numpy coordinates are [z, y, x] and SimpleITK's are [x, y, z]
            x_cropped = np.moveaxis(x_cropped, [0, 2], [2, 0])

            # Changing the study name if needed
            if filename in files_with_swapped_masks:
                study_name = match[0] + ("_R" if side == "_L" else "_L")

            # Save new images so they can be loaded directly
            sitk.WriteImage(sitk.GetImageFromArray(y),
                            join_path([target_path, study_name + "_gt.nii.gz"]))
            nib.save(nib.Nifti1Image(x_cropped, affine),
                     join_path([target_path, study_name + ".nii.gz"]))
