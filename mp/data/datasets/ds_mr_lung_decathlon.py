# ------------------------------------------------------------------------------
# Lung task from the Medical Segmentation Decathlon 
# (http://medicaldecathlon.com/)
# ------------------------------------------------------------------------------

# Necessary imports
import os
import numpy as np
import torch
import SimpleITK as sitk
from mp.data.pytorch.transformation import centre_crop_pad_2d
import random
from mp.utils.load_restore import join_path
from mp.data.datasets.dataset_regression import RegressionDataset, RegressionInstance
from mp.paths import storage_data_path
import mp.data.datasets.dataset_utils as du
from mp.data.datasets.dataset_augmentation import augment_data_in_four_intensities as augment_data
from mp.data.datasets.dataset_augmentation import select_random_images_slices, save_dataset

class DecathlonLung(RegressionDataset):
    r"""Class for the Lung decathlon challenge, contains only
    CT, found at http://medicaldecathlon.com/.
    """
    def __init__(self, subset=None, hold_out_ixs=[], augmented=False,
        img_size=(1, 256, 256), max_likert_value=1, random_slices=False,
        noise='blur', nr_images=200, nr_slices=20, original_perc_data=0.2):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths    
        global_name = 'DecathlonLung'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)
        folder_name = 'randomised_data_' + str(noise)   # For random selected data

        # Extract all images, if not already done
        if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
            _extract_images(original_data_path, dataset_path, img_size)

        if random_slices:
            t_path = os.path.join(dataset_path, folder_name)
            _extract_images_random(dataset_path, global_name,
                                   folder_name, nr_images,
                                   nr_slices, original_perc_data,
                                   storage_data_path)
            
            # Fetch all random patient/study names that do not begin with '._'
            study_names_random = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                in os.listdir(t_path) if '._' not in file_name and 'lung' in file_name)

        # Fetch all patient/study names that do not begin with '._'
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
            in os.listdir(dataset_path) if '._' not in file_name and 'lung' in file_name)
            
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
            labels, names = augment_data(instances_full, 'DecathlonLungAugmented',
                                                   True, False, storage_data_path,
                                                   max_likert_value, random_slices,
                                                   noise, nr_images, nr_slices)

            # Add to instances
            if random_slices:
                for num, name in enumerate(names):
                    msg = 'Creating dataset from random SimpleITK images and slices: '
                    msg += str(num + 1) + ' of ' + str(len(names)) + '.'
                    print (msg, end = '\r')
                    instances.append(RegressionInstance(
                        x_path=os.path.join(storage_data_path, 'DecathlonLungAugmented',
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
                        x_path=os.path.join(storage_data_path, 'DecathlonLungAugmented',
                                            'augmented_data', name+'.nii.gz'),
                        y_label=labels[name],
                        name=name,
                        group_id=None
                        ))

        super().__init__(instances, name=global_name,
            modality='CT', nr_channels=1, hold_out_ixs=[])

def _extract_images(source_path, target_path, img_size=(1, 256, 256)):
    r"""Extracts MRI images and saves the modified images."""
    images_path = os.path.join(source_path, 'imagesTr')

    # Filenames have the form 'lung_XXX.nii.gz'
    filenames = [x for x in os.listdir(images_path) if x[:4] == 'lung']

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
                           nr_images, nr_slices, original_perc_data,
                           storage_data_path=storage_data_path):
    r"""Extracts MRI images and slices randomly based on input and saves
        the images."""

    # Calculate number of images to select
    nr_images = int(nr_images*original_perc_data)
    # Extract filenames
    filenames = [x for x in os.listdir(source_path) if '._' not in x]
    # Define noise, in this case it is just a string contained in the filenames
    noise = 'lung'
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