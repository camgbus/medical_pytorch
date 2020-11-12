# ------------------------------------------------------------------------------
# Hippocampus segmentation task for the HarP dataset
# (http://www.hippocampal-protocol.net/SOPs/index.php)
# ------------------------------------------------------------------------------

import os

import SimpleITK as sitk
import skimage.transform as transform

import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
from mp.utils.load_restore import join_path


class MontgomeryJSRT(SegmentationDataset):
    r"""Class for the segmentation of the Montgomery and JSRT datasets
    as provided by https://data.gov.au/dataset/ds-dap-csiro%3A34717/details?q=
    The Montgomery datset has been augmented to match the image size of the JSRT dataset
    """

    def __init__(self, subset=None, hold_out_ixs=None, resize=True):
        # Part is either: "Montgomery" or "JSRT"
        default = {"Dataset": "Montgomery"}
        if subset is not None:
            default.update(subset)
            subset = default
        else:
            subset = default

        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = subset["Dataset"]
        name = du.get_dataset_name(global_name, subset)
        dataset_path = os.path.join(storage_data_path, global_name, "Resized" if resize else "Original")
        original_data_path = du.get_original_data_path("MontgomeryJSRT")

        # Copy the images if not done already
        if not os.path.isdir(dataset_path):
            _extract_images(original_data_path,
                            dataset_path,
                            "JPCNN" if subset["Dataset"] == "JSRT" else "MCUCXR_",
                            resize)

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
                         modality='CXR', nr_channels=1, hold_out_ixs=hold_out_ixs)


def _extract_images(source_path, target_path, file_prefix, resize):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """
    images_path = os.path.join(source_path, 'Images')
    labels_path = os.path.join(source_path, 'Masks')

    filenames = [x for x in os.listdir(labels_path) if x.startswith(file_prefix)]

    # Create directories
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for filename in filenames:
        # No specific processing
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = sitk.GetArrayFromImage(x)
        y = sitk.ReadImage(os.path.join(labels_path, filename))
        y = sitk.GetArrayFromImage(y)
        # Shape expected: (2048, 2048)
        assert x.shape == y.shape

        if resize:
            x = transform.resize(x, (512, 512), anti_aliasing=True)
            y = transform.resize(y, (512, 512), anti_aliasing=False)

        # Save new images so they can be loaded directly
        study_name = filename.replace(file_prefix, '').split('.tif')[0]
        sitk.WriteImage(sitk.GetImageFromArray(x), join_path([target_path, study_name + ".nii.gz"]))
        sitk.WriteImage(sitk.GetImageFromArray(y), join_path([target_path, study_name + "_gt.nii.gz"]))
