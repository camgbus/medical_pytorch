# ------------------------------------------------------------------------------
# Hippocampus segmentation published by Dryad
# (https://datadryad.org/stash/dataset/doi:10.5061/dryad.gc72v)
# ------------------------------------------------------------------------------

import os
import re

import SimpleITK as sitk

import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
from mp.utils.load_restore import join_path


class DryadHippocampus(SegmentationDataset):
    r"""Class for the segmentation of the HarP dataset,
    https://datadryad.org/stash/dataset/doi:10.5061/dryad.gc72v.
    """

    def __init__(self, subset=None, hold_out_ixs=None, merge_labels=True):
        # Modality is either: "T1w" or "T2w"
        # Resolution is either: "Standard" or "Hires"
        # If you want to use different resolutions or modalities, please create another object with a different subset
        # Currently only standard resolution is implemented
        default = {"Modality": "T1w", "Resolution": "Standard"}
        if subset is not None:
            default.update(subset)
            subset = default
        else:
            subset = default

        # Standard resolution T2w is not available
        assert not (subset["Resolution"] == "Standard" and subset["Modality"] == "T2w"), \
            "Standard resolution T2w not available for the Dryad Hippocampus dataset"

        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = 'DryadHippocampus'
        name = du.get_dataset_name(global_name, subset)
        dataset_path = os.path.join(storage_data_path,
                                    global_name,
                                    "Merged Labels" if merge_labels else "Original",
                                    "".join([f"{key}[{subset[key]}]" for key in ["Modality", "Resolution"]])
                                    )
        original_data_path = du.get_original_data_path(global_name)

        # Copy the images if not done already
        if not os.path.isdir(dataset_path):
            _extract_images(original_data_path, dataset_path, merge_labels, subset)

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

        if merge_labels:
            label_names = ['background', 'hippocampus']
        else:
            label_names = ['background', 'subiculum', 'CA1-3', 'CA4-DG']

        super().__init__(instances, name=name, label_names=label_names,
                         modality=subset["Modality"] + ' MRI', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path, merge_labels, subset):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    # Create directories
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    # Patient folders s01, s02, ...
    for patient_folder in filter(lambda s: re.match(r"^s[0-9]+.*", s), os.listdir(source_path)):

        # Loading the image
        image_path = os.path.join(source_path, patient_folder,
                                  f"{patient_folder}_{subset['Modality'].lower()}_"
                                  f"{subset['Resolution'].lower()}_defaced_MNI.nii.gz")
        x = sitk.ReadImage(image_path)
        x = sitk.GetArrayFromImage(x)

        # For each MRI, there are 2 segmentation (left and right hippocampus)
        for side in ("L", "R"):
            # Loading the label
            label_path = os.path.join(source_path, patient_folder,
                                      f"{patient_folder}_hippolabels_"
                                      f"{'hres' if subset['Resolution'] == 'Hires' else 't1w_standard'}"
                                      f"_{side}_MNI.nii.gz")

            y = sitk.ReadImage(label_path)
            y = sitk.GetArrayFromImage(y)

            # We need to recover the study name of the image name to construct the name of the segmentation files
            study_name = f"{patient_folder}_{side}"

            # Shape expected: (189, 233, 197)
            assert x.shape == y.shape

            # Cropping bounds computed to fit the ground truth
            if subset["Resolution"] == "Standard":
                if side == "L":
                    y = y[40: 104, 78: 142, 49: 97]
                    x_cropped = x[40: 104, 78: 142, 49: 97]
                else:
                    y = y[40: 104, 78: 142, 97: 145]
                    x_cropped = x[40: 104, 78: 142, 97: 145]
            else:
                if side == "L":
                    y = y[88: 216, 190: 318, 81: 209]
                    x_cropped = x[88: 216, 190: 318, 81: 209]
                else:
                    y = y[88: 216, 190: 318, 218: 346]
                    x_cropped = x[88: 216, 190: 318, 218: 346]

            if merge_labels:
                y[y > 1] = 1

            # Save new images so they can be loaded directly
            sitk.WriteImage(sitk.GetImageFromArray(y),
                            join_path([target_path, study_name + "_gt.nii.gz"]))
            sitk.WriteImage(sitk.GetImageFromArray(x_cropped),
                            join_path([target_path, study_name + ".nii.gz"]))
