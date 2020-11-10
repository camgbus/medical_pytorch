# ------------------------------------------------------------------------------
# Hippocampus segmentation task from the Medical Segmentation Decathlon
# (http://medicaldecathlon.com/)
# ------------------------------------------------------------------------------

import os

import SimpleITK as sitk

import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
from mp.utils.load_restore import join_path
import matplotlib.pyplot as plt


class Shenzhen(SegmentationDataset):
    r"""Class for the Shenzhen lung segmentation dataset,
    found at https://lhncbc.nlm.nih.gov/publication/pub9931
    and for masks https://www.kaggle.com/yoctoman/shcxr-lung-mask.
    """

    def __init__(self, subset=None, hold_out_ixs=None, merge_labels=True):
        assert subset is None, "No subsets for this dataset."

        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = 'Shenzhen'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        # Copy the images if not done already
        import shutil
        shutil.rmtree(dataset_path)
        if not os.path.isdir(dataset_path):
            _extract_images(original_data_path, dataset_path, merge_labels)

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

        label_names = ['background', 'lung']
        super().__init__(instances, name=global_name, label_names=label_names,
                         modality='CXT', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path, merge_labels):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    images_path = os.path.join(source_path, 'CXR_png')
    labels_path = os.path.join(source_path, 'mask')

    filenames = [x for x in os.listdir(labels_path)]

    # Create directories
    os.makedirs(target_path)

    # There are some masks missing
    for filename in filenames:
        # Extract only T2-weighted
        # pyplot reads the scan as RGB but not the label, and we just need to add a channel axis
        x = plt.imread(os.path.join(images_path, filename.replace("_mask", "")))[..., 0][..., None]
        y = plt.imread(os.path.join(labels_path, filename))[..., None]
        print(x.shape)
        assert x.shape == y.shape

        # Save new images so they can be loaded directly
        study_name = filename.replace("_mask", "").split('.png')[0]
        sitk.WriteImage(sitk.GetImageFromArray(x), join_path([target_path, study_name + ".nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(y), join_path([target_path, study_name + "_gt.nii.gz"]))
        break