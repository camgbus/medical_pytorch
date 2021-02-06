# ------------------------------------------------------------------------------
# Grand Challenge Corona Datset: https://covid-segmentation.grand-challenge.org.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import numpy as np
import torch
import json
import SimpleITK as sitk
from mp.data.pytorch.transformation import centre_crop_pad_2d
import random
from mp.utils.load_restore import join_path
from mp.data.datasets.dataset_regression import RegressionDataset, RegressionInstance
from mp.paths import storage_data_path
import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_augmentation import augment_data_in_four_intensities as augment_data
from mp.data.datasets.dataset_augmentation import select_random_images_slices, save_dataset

class GCCorona(RegressionDataset):
    r"""Class for the Grand Challenge Corona data, contains only
    CT, found at https://covid-segmentation.grand-challenge.org/.
    """
    def __init__(self, subset=None, hold_out_ixs=[], augmented=False,
        img_size=(1, 299, 299), max_likert_value=1, random_slices=False,
        noise='blur', nr_images=200, nr_slices=20):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths    
        global_name = 'GC_Corona'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)
        folder_name = 'randomised_data_regression_' + str(noise)   # For random selected data

        # Extract all images, if not already done
        if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
            _extract_images(original_data_path, dataset_path, img_size)

        if random_slices:
            t_path = os.path.join(dataset_path, folder_name)
            _extract_images_random(dataset_path, global_name,
                                   folder_name, nr_images,
                                   nr_slices,
                                   storage_data_path)
            
            # Fetch all random patient/study names that do not begin with '._'
            study_names_random = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                in os.listdir(t_path) if '._' not in file_name and 'covid19' in file_name
                and '_seg' not in file_name)

        # Fetch all patient/study names that do not begin with '._'
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
            in os.listdir(dataset_path) if '._' not in file_name and 'covid19' in file_name
            and '_seg' not in file_name)
            
        # Build instances
        instances = []
        instances_full = []
        # Load all data into instances_full, needed to augment all data instances once
        for num, study_name in enumerate(study_names):
            msg = 'Creating dataset from SimpleITK images: '
            msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
            print (msg, end = '\r')
            instances_full.append(RegressionInstance(
                x_path=os.path.join(dataset_path, study_name+'.nii.gz'),
                y_label=torch.tensor([1/max_likert_value]),
                name=study_name,
                group_id=None
                ))

        if random_slices:
            for num, study_name in enumerate(study_names_random):
                msg = 'Creating dataset from random SimpleITK images and slices: '
                msg += str(num + 1) + ' of ' + str(len(study_names_random)) + '.'
                print (msg, end = '\r')
                instances.append(RegressionInstance(
                    x_path=os.path.join(t_path,
                                        study_name+'.nii.gz'),
                    y_label=torch.tensor([1/max_likert_value]),
                    name=study_name,
                    group_id=None
                    ))
        else:
            instances = instances_full
                    
        # Create augmented images and add them if it is desired
        if augmented:
            labels, names = augment_data(instances_full, 'GC_CoronaAugmented',
                                                   True, False, storage_data_path,
                                                   max_likert_value, random_slices,
                                                   noise, nr_images, nr_slices, 'regression')

            # Add to instances
            if random_slices:
                for num, name in enumerate(names):
                    msg = 'Creating dataset from random SimpleITK images and slices: '
                    msg += str(num + 1) + ' of ' + str(len(names)) + '.'
                    print (msg, end = '\r')
                    instances.append(RegressionInstance(
                        x_path=os.path.join(storage_data_path, 'GC_CoronaAugmented',
                                            folder_name, name+'.nii.gz'),
                        y_label=labels[name],
                        name=name,
                        group_id=None
                        ))
            else:
                for num, name in enumerate(names):
                    msg = 'Creating dataset from augmented SimpleITK images: '
                    msg += str(num + 1) + ' of ' + str(len(names)) + '.'
                    print (msg, end = '\r')
                    instances.append(RegressionInstance(
                        x_path=os.path.join(storage_data_path, 'GC_CoronaAugmented',
                                            'augmented_data', name+'.nii.gz'),
                        y_label=labels[name],
                        name=name,
                        group_id=None
                        ))

        super().__init__(instances, name=global_name,
            modality='CT', nr_channels=1, hold_out_ixs=[])

class GCCoronaRestored(RegressionDataset):
    r"""Class for the Grand Challenge Corona data, contains only
    CT, found at https://covid-segmentation.grand-challenge.org/.
    This class is used to train a restored model with the same
    data, e.g. if the training interrupted due to an error. It is
    important that the original images and random image folders
    (GC_Corona/randomised_data_cnn_<noise> and
    GC_CoronaAugmented/randomised_data_cnn_<noise>) exists and are not
    empty. Further the corresponding Labels, creating by augmenting images
    or performing training for first time need to present at the same location
    as created (GC_CoronaAugmented/labels/labels.json).
    """
    def __init__(self, subset=None, hold_out_ixs=[], img_size=(1, 299, 299),
        max_likert_value=1, noise='blur'):

        # Extract necessary paths
        global_name = 'GC_Corona'
        dataset_path = os.path.join(storage_data_path, global_name)
        random_path = os.path.join(storage_data_path, global_name+'Augmented')
        original_data_path = du.get_original_data_path(global_name)
        folder_name = 'randomised_data_regression_' + str(noise)
        t_path = os.path.join(dataset_path, folder_name)
        r_path = os.path.join(random_path, folder_name)

        # Fetch all patient/study names that do not begin with '._' for random and original images
        study_names_random_orig = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                        in os.listdir(t_path) if '._' not in file_name and 'covid19' in file_name
                        and '_seg' not in file_name)
        study_names_random_augm = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                        in os.listdir(r_path) if '._' not in file_name and 'covid19' in file_name
                        and '_seg' not in file_name)

        # Load labels
        with open(os.path.join(storage_data_path,
        global_name+'Augmented', 'labels', 'labels.json'), 'r') as fp:
            labels = json.load(fp)

        # Transform label integers into torch.tensors
        for key, value in labels.items():
            labels[key] = torch.tensor([value])

        # Build instances
        instances = []
        # Add image path and labels to instances
        for num, study_name in enumerate(study_names_random_orig):
            msg = 'Creating dataset from random SimpleITK images and slices: '
            msg += str(num + 1) + ' of ' + str(len(study_names_random_orig)) + '.'
            print (msg, end = '\r')
            instances.append(RegressionInstance(
                x_path=os.path.join(t_path,
                                    study_name+'.nii.gz'),
                y_label=torch.tensor([1/max_likert_value]),
                name=study_name,
                group_id=None
                ))

        for num, study_name in enumerate(study_names_random_augm):
            msg = 'Creating dataset from random SimpleITK images and slices: '
            msg += str(num + 1) + ' of ' + str(len(study_names_random_augm)) + '.'
            print (msg, end = '\r')
            instances.append(RegressionInstance(
                x_path=os.path.join(r_path,
                                    study_name+'.nii.gz'),
                y_label=labels[study_name],
                name=study_name,
                group_id=None
                ))

        super().__init__(instances, name=global_name,
                    modality='CT', nr_channels=1, hold_out_ixs=[])


def _extract_images(source_path, target_path, img_size=(1, 299, 299)):
    r"""Extracts MRI images and saves the modified images."""
    images_path = os.path.join(source_path, 'Train')

    # Filenames have the form 'volume-covid19-A-XXXX_ct.nii'
    filenames = [x for x in os.listdir(images_path) if 'covid19' in x
                 and '_seg' not in x and '._' not in x]

    # Create directories if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for num, filename in enumerate(filenames):
        msg = "Loading SimpleITK images and resizing them: "
        msg += str(num + 1) + " of " + str(len(filenames)) + "."
        print (msg, end = "\r")
        # Extract all images (3D)
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = torch.from_numpy(sitk.GetArrayFromImage(x))
        try:
            x = centre_crop_pad_2d(x, img_size)
        except:
            print('Image could not be resized and will therefore be skipped: {}.'
            .format(filename))
            continue
        # Save new images so they can be loaded directly
        study_name = filename.replace('_', '').split('.nii')[0]
        sitk.WriteImage(sitk.GetImageFromArray(x), 
            join_path([target_path, study_name+".nii.gz"]))

def _extract_images_random(source_path, data_label, folder_name,
                           nr_images, nr_slices,
                           storage_data_path=storage_data_path):
    r"""Extracts MRI images and slices randomly based on input and saves
        the images."""
    # Extract filenames
    filenames = [x for x in os.listdir(source_path) if '._' not in x]
    # Define noise, in this case it is just a string contained in the filenames
    noise = 'covid19'
    # Select random images based on nr_images and random slices
    # for each image based on nr_slices
    random_data, image_names = select_random_images_slices(source_path, filenames, noise,
                                                           nr_images, nr_slices)
    # Save random images so they can be loaded
    print('Saving random images and image slices as SimpleITK for training and testing..')
    save_dataset(random_data,
                 image_names,
                 data_label,
                 folder_name,
                 storage_data_path,
                 simpleITK=True,
                 empty_dir=True)

def _generate_labels(target_path, max_likert_value):
    r"""Generates the labels based on real assessment and saves them into
    target_path with corresponding names."""
    print("Generating labels..")
    labels = dict()

    labels['volume-covid19-A-0003_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0003_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0003_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0003_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0003_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0003_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0011_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0011_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0011_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0011_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0011_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0011_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0013_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0013_ct'+'_downsample'] = 2/max_likert_value
    labels['volume-covid19-A-0013_ct'+'_ghosting'] = 2/max_likert_value
    labels['volume-covid19-A-0013_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0013_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0013_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0014_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0014_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0014_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0014_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0014_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0014_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0016_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0016_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0016_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0016_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0016_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0016_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0025_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0025_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0025_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0025_ct'+'_motion'] = 5/max_likert_value
    labels['volume-covid19-A-0025_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0025_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0031_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0031_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0031_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0031_ct'+'_motion'] = 4/max_likert_value
    labels['volume-covid19-A-0031_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0031_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0034_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0034_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0034_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0034_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0034_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0034_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0038_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0038_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0038_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0038_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0038_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0038_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0039_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0039_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0039_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0039_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0039_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0039_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0041_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0041_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0041_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0041_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0041_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0041_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0044_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0044_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0044_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0044_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0044_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0044_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0046_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0046_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0046_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0046_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0046_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0046_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0047_1_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0047_1_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0047_1_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0047_1_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0047_1_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0047_1_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0053_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0053_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0053_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0053_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0053_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0053_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0054_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0054_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0054_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0054_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0054_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0054_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0066_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0066_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0066_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0066_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0066_ct'+'_noise'] = 2/max_likert_value
    labels['volume-covid19-A-0066_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0070_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0070_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0070_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0070_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0070_ct'+'_noise'] = 2/max_likert_value
    labels['volume-covid19-A-0070_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0072_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0072_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0072_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0072_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0072_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0072_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0073_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0073_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0073_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0073_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0073_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0073_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0074_1_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0074_1_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0074_1_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0074_1_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0074_1_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0074_1_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0077_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0077_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0077_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0077_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0077_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0077_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0083_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0083_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0083_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0083_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0083_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0083_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0090_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0090_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0090_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0090_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0090_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0090_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0092_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0092_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0092_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0092_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0092_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0092_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0096_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0096_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0096_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0096_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0096_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0096_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0106_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0106_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0106_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0106_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0106_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0106_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0110_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0110_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0110_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0110_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0110_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0110_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0112_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0112_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0112_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0112_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0112_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0112_ct'+'_spike'] = 3/max_likert_value
    labels['volume-covid19-A-0114_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0114_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0114_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0114_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0114_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0114_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0120_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0120_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0120_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0120_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0120_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0120_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0129_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0129_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0129_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0129_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0129_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0129_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0130_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0130_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0130_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0130_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0130_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0130_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0133_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0133_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0133_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0133_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0133_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0133_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0147_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0147_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0147_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0147_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0147_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0147_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0151_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0151_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0151_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0151_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0151_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0151_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0154_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0154_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0154_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0154_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0154_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0154_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0161_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0161_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0161_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0161_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0161_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0161_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0164_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0164_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0164_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0164_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0164_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0164_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0165_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0165_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0165_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0165_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0165_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0165_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0167_1_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0167_1_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0167_1_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0167_1_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0167_1_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0167_1_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0173_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0173_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0173_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0173_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0173_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0173_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0178_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0178_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0178_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0178_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0178_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0178_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0179_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0179_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0179_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0179_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0179_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0179_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0181_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0181_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0181_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0181_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0181_ct'+'_noise'] = 2/max_likert_value
    labels['volume-covid19-A-0181_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0187_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0187_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0187_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0187_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0187_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0187_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0196_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0196_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0196_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0196_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0196_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0196_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0199_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0199_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0199_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0199_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0199_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0199_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0201_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0201_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0201_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0201_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0201_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0201_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0202_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0202_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0202_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0202_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0202_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0202_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0214_ct'+'_blur'] = 4/max_likert_value
    labels['volume-covid19-A-0214_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0214_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0214_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0214_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0214_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0215_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0215_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0215_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0215_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0215_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0215_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0228_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0228_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0228_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0228_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0228_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0228_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0233_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0233_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0233_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0233_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0233_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0233_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0236_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0236_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0236_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0236_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0236_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0236_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0237_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0237_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0237_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0237_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0237_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0237_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0239_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0239_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0239_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0239_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0239_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0239_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0240_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0240_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0240_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0240_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0240_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0240_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0246_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0246_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0246_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0246_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0246_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0246_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0247_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0247_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0247_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0247_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0247_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0247_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0251_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0251_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0251_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0251_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0251_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0251_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0252_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0252_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0252_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0252_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0252_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0252_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0255_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0255_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0255_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0255_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0255_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0255_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0256_1_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0256_1_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0256_1_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0256_1_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0256_1_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0256_1_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0263_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0263_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0263_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0263_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0263_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0263_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0264_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0264_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0264_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0264_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0264_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0264_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0267_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0267_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0267_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0267_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0267_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0267_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0270_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0270_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0270_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0270_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0270_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0270_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0282_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0282_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0282_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0282_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0282_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0282_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0285_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0285_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0285_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0285_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0285_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0285_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0288_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0288_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0288_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0288_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0288_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0288_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0295_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0295_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0295_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0295_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0295_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0295_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0296_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0296_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0296_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0296_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0296_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0296_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0299_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0299_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0299_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0299_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0299_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0299_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0301_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0301_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0301_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0301_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0301_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0301_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0307_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0307_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0307_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0307_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0307_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0307_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0313_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0313_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0313_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0313_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0313_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0313_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0314_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0314_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0314_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0314_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0314_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0314_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0315_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0315_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0315_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0315_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0315_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0315_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0316_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0316_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0316_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0316_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0316_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0316_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0319_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0319_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0319_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0319_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0319_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0319_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0320_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0320_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0320_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0320_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0320_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0320_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0323_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0323_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0323_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0323_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0323_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0323_ct'+'_spike'] = 3/max_likert_value
    labels['volume-covid19-A-0329_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0329_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0329_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0329_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0329_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0329_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0331_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0331_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0331_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0331_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0331_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0331_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0332_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0332_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0332_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0332_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0332_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0332_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0338_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0338_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0338_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0338_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0338_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0338_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0339_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0339_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0339_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0339_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0339_ct'+'_noise'] = 2/max_likert_value
    labels['volume-covid19-A-0339_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0342_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0342_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0342_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0342_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0342_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0342_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0347_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0347_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0347_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0347_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0347_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0347_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0351_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0351_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0351_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0351_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0351_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0351_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0354_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0354_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0354_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0354_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0354_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0354_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0355_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0355_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0355_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0355_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0355_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0355_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0360_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0360_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0360_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0360_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0360_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0360_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0361_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0361_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0361_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0361_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0361_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0361_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0366_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0366_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0366_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0366_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0366_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0366_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0372_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0372_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0372_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0372_ct'+'_motion'] = 4/max_likert_value
    labels['volume-covid19-A-0372_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0372_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0377_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0377_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0377_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0377_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0377_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0377_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0380_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0380_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0380_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0380_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0380_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0380_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0382_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0382_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0382_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0382_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0382_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0382_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0383_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0383_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0383_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0383_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0383_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0383_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0386_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0386_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0386_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0386_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0386_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0386_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0387_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0387_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0387_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0387_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0387_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0387_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0388_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0388_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0388_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0388_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0388_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0388_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0391_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0391_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0391_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0391_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0391_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0391_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0392_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0392_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0392_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0392_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0392_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0392_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0394_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0394_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0394_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0394_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0394_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0394_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0397_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0397_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0397_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0397_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0397_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0397_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0400_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0400_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0400_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0400_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0400_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0400_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0402_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0402_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0402_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0402_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0402_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0402_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0407_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0407_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0407_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0407_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0407_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0407_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0413_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0413_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0413_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0413_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0413_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0413_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0414_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0414_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0414_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0414_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0414_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0414_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0416_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0416_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0416_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0416_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0416_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0416_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0417_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0417_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0417_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0417_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0417_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0417_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0418_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0418_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0418_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0418_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0418_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0418_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0421_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0421_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0421_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0421_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0421_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0421_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0422_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0422_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0422_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0422_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0422_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0422_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0423_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0423_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0423_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0423_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0423_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0423_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0435_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0435_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0435_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0435_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0435_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0435_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0437_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0437_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0437_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0437_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0437_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0437_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0443_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0443_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0443_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0443_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0443_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0443_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0445_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0445_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0445_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0445_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0445_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0445_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0455_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0455_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0455_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0455_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0455_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0455_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0462_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0462_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0462_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0462_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0462_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0462_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0463_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0463_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0463_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0463_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0463_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0463_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0464_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0464_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0464_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0464_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0464_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0464_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0472_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0472_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0472_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0472_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0472_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0472_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0473_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0473_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0473_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0473_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0473_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0473_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0475_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0475_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0475_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0475_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0475_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0475_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0476_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0476_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0476_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0476_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0476_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0476_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0479_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0479_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0479_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0479_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0479_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0479_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0483_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0483_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0483_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0483_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0483_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0483_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0494_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0494_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0494_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0494_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0494_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0494_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0495_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0495_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0495_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0495_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0495_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0495_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0498_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0498_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0498_ct'+'_ghosting'] = 2/max_likert_value
    labels['volume-covid19-A-0498_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0498_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0498_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0500_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0500_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0500_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0500_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0500_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0500_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0502_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0502_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0502_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0502_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0502_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0502_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0504_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0504_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0504_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0504_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0504_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0504_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0511_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0511_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0511_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0511_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0511_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0511_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0521_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0521_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0521_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0521_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0521_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0521_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0522_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0522_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0522_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0522_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0522_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0522_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0524_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0524_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0524_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0524_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0524_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0524_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0525_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0525_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0525_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0525_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0525_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0525_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0526_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0526_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0526_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0526_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0526_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0526_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0530_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0530_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0530_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0530_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0530_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0530_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0531_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0531_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0531_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0531_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0531_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0531_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0534_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0534_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0534_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0534_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0534_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0534_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0537_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0537_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0537_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0537_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0537_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0537_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0547_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0547_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0547_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0547_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0547_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0547_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0548_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0548_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0548_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0548_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0548_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0548_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0553_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0553_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0553_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0553_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0553_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0553_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0557_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0557_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0557_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0557_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0557_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0557_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0559_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0559_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0559_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0559_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0559_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0559_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0560_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0560_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0560_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0560_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0560_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0560_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0562_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0562_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0562_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0562_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0562_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0562_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0566_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0566_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0566_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0566_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0566_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0566_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0567_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0567_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0567_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0567_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0567_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0567_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0569_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0569_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0569_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0569_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0569_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0569_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0570_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0570_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0570_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0570_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0570_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0570_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0573_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0573_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0573_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0573_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0573_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0573_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0575_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0575_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0575_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0575_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0575_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0575_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0576_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0576_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0576_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0576_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0576_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0576_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0579_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0579_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0579_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0579_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0579_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0579_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0581_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0581_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0581_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0581_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0581_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0581_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0585_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0585_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0585_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0585_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0585_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0585_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0586_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0586_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0586_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0586_ct'+'_motion'] = 3/max_likert_value
    labels['volume-covid19-A-0586_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0586_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0589_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0589_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0589_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0589_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0589_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0589_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0590_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0590_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0590_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0590_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0590_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0590_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0599_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0599_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0599_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0599_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0599_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0599_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0604_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0604_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0604_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0604_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0604_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0604_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0612_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0612_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0612_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0612_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0612_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0612_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0614_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0614_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0614_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0614_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0614_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0614_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0623_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0623_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0623_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0623_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0623_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0623_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0626_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0626_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0626_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0626_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0626_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0626_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0627_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0627_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0627_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0627_ct'+'_motion'] = 4/max_likert_value
    labels['volume-covid19-A-0627_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0627_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0629_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0629_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0629_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0629_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0629_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0629_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0635_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0635_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0635_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0635_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0635_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0635_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0636_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0636_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0636_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0636_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0636_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0636_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0638_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0638_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0638_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0638_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0638_ct'+'_noise'] = 3/max_likert_value
    labels['volume-covid19-A-0638_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0643_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0643_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0643_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0643_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0643_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0643_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0648_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0648_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0648_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0648_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0648_ct'+'_noise'] = 5/max_likert_value
    labels['volume-covid19-A-0648_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0652_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0652_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0652_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0652_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0652_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0652_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0656_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0656_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0656_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0656_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0656_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0656_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0657_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0657_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0657_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0657_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0657_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0657_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0658_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0658_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0658_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0658_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0658_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0658_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0659_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0659_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0659_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0659_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0659_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0659_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0660_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0660_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0660_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0660_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0660_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0660_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0665_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0665_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0665_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0665_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0665_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0665_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0666_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0666_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0666_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0666_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0666_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0666_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0669_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0669_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0669_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0669_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0669_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0669_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0670_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0670_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0670_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0670_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0670_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0670_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0678_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0678_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0678_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0678_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0678_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0678_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0685_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0685_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0685_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0685_ct'+'_motion'] = 2/max_likert_value
    labels['volume-covid19-A-0685_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0685_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0686_ct'+'_blur'] = 2/max_likert_value
    labels['volume-covid19-A-0686_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0686_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0686_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0686_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0686_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0694_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0694_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0694_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0694_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0694_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0694_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_blur'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_noise'] = 4/max_likert_value
    labels['volume-covid19-A-0600_ct'+'_spike'] = 1/max_likert_value
    labels['volume-covid19-A-0698_ct'+'_blur'] = 3/max_likert_value
    labels['volume-covid19-A-0698_ct'+'_downsample'] = 1/max_likert_value
    labels['volume-covid19-A-0698_ct'+'_ghosting'] = 1/max_likert_value
    labels['volume-covid19-A-0698_ct'+'_motion'] = 1/max_likert_value
    labels['volume-covid19-A-0698_ct'+'_noise'] = 1/max_likert_value
    labels['volume-covid19-A-0698_ct'+'_spike'] = 1/max_likert_value
    
    # Save the labels
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    with open(os.path.join(target_path, 'labels.json'), 'w') as fp:
        json.dump(labels, fp, sort_keys=True, indent=4)