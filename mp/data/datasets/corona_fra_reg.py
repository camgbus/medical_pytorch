# ------------------------------------------------------------------------------
# Corona dataset from Uniklinik Frankfurt
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

class FraCoronaDatasetAugmented(RegressionDataset):
    r"""Class for the Corona dataset provided by the Uniklinik Frankfurt.
    """
    def __init__(self, subset=None, hold_out_ixs=[], augmented=False,
        img_size=(1, 299, 299), max_likert_value=1, random_slices=False,
        noise='blur', nr_images=20, nr_slices=20, set_name='train'):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths    
        global_name = 'FRACorona'
        dataset_path = os.path.join(storage_data_path, global_name, set_name)
        original_data_path = os.path.join(du.get_original_data_path(global_name), set_name)
        folder_name = 'randomised_data_regression_' + str(noise)   # For random selected data

        # Extract all images, if not already done
        if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
            _extract_images(original_data_path, dataset_path, img_size)

        if random_slices:
            t_path = os.path.join(dataset_path, folder_name)
            _extract_images_random(dataset_path, global_name,
                                   set_name+'/'+folder_name, nr_images,
                                   nr_slices,
                                   storage_data_path)
            
            # Fetch all random patient/study names that do not begin with '._'
            study_names_random = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                in os.listdir(t_path) if '._' not in file_name and 'KGU' in file_name)

        # Fetch all patient/study names that do not begin with '._'
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
            in os.listdir(dataset_path) if '._' not in file_name and 'KGU' in file_name)
            
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
            labels, names = augment_data(instances_full, 'FRACoronaAugmented',
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
                        x_path=os.path.join(storage_data_path, 'FRACoronaAugmented',
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
                        x_path=os.path.join(storage_data_path, 'FRACoronaAugmented',
                                            'augmented_data', name+'.nii.gz'),
                        y_label=labels[name],
                        name=name,
                        group_id=None
                        ))

        super().__init__(instances, name=global_name,
            modality='CT', nr_channels=1, hold_out_ixs=[])


class FraCoronaDataset(RegressionDataset):
    r"""Class for the Corona dataset provided by the Uniklinik Frankfurt.
    """
    def __init__(self, subset=None, hold_out_ixs=[], augmented=False,
        img_size=(1, 299, 299), max_likert_value=1, noise='blur', set_name='train'):
        assert subset is None, "No subsets for this dataset."

        # Extract necessary paths    
        global_name = 'FRACorona'
        dataset_path = os.path.join(storage_data_path, global_name, set_name)
        original_data_path = os.path.join(du.get_original_data_path(global_name), set_name)

        # Fetch all patient/study names that do not begin with '._'
        study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
            in os.listdir(dataset_path) if '._' not in file_name and 'KGU' in file_name)
            
        # Create all label files, if not already done
        if not os.path.isdir(os.path.join(dataset_path, 'labels'))\
           or not os.listdir(os.path.join(dataset_path, 'labels')):
            _generate_labels(dataset_path, max_likert_value)

        # Load labels (need to be added by hand in folder 
        # FRACorona/labels as labels.json!)
        labels = dict()
        with open(os.path.join(dataset_path, 'labels', 'labels'+str(noise)+'.json'), 'r') as fp:
            labels = json.load(fp)

        # Transform label integers into torch.tensors
        for key, value in labels.items():
            labels[key] = torch.tensor([value])

        # Build instances
        instances = []
        
        # Load all data into instances
        for num, study_name in enumerate(study_names):
            msg = 'Creating dataset from SimpleITK images: '
            msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
            print (msg, end = '\r')
            instances.append(RegressionInstance(
                x_path=os.path.join(dataset_path, study_name+'.nii.gz'),
                y_label=labels[study_name],
                name=study_name,
                group_id=None
                ))

        super().__init__(instances, name=global_name,
            modality='CT', nr_channels=1, hold_out_ixs=[])


class FraCoronaDatasetRestored(RegressionDataset):
    r"""Class for the Corona dataset provided by the Uniklinik Frankfurt. This class is used
    to train a restored model with the same data, e.g. if the training
    interrupted due to an error. It is important that the original
    images and random image folders (FRACorona/randomised_data_regression_<noise>
    and FRACoronaAugmented/randomised_data_regression_<noise>) exists and are not
    empty. Further the corresponding Labels, creating by augmenting images
    or performing training for first time need to present at the same location
    as created (FRACoronaAugmented/labels/labels.json).
    """
    def __init__(self, subset=None, hold_out_ixs=[], img_size=(1, 299, 299),
        max_likert_value=1, noise='blur', set_name='test'):

        # Extract necessary paths
        global_name = 'FRACorona'
        dataset_path = os.path.join(storage_data_path, global_name, set_name)
        random_path = os.path.join(storage_data_path, global_name+'Augmented')
        original_data_path = os.path.join(du.get_original_data_path(global_name), set_name)
        folder_name = 'randomised_data_regression_' + str(noise)
        t_path = os.path.join(dataset_path, folder_name)
        r_path = os.path.join(random_path, folder_name)

        # Fetch all patient/study names that do not begin with '._' for random and original images
        study_names_random_orig = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                        in os.listdir(t_path) if '._' not in file_name and 'KGU' in file_name)
        study_names_random_augm = set(file_name.split('.nii')[0].split('_gt')[0] for file_name 
                        in os.listdir(r_path) if '._' not in file_name and 'KGU' in file_name)

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
    images_path = source_path

    # Filenames are provided in foldernames: ptient_id/images.nii.gz
    filenames = set(file_name for file_name in os.listdir(images_path)
                    if file_name[:1] != '.')

    # Create directories if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for num, filename in enumerate(filenames):
        msg = "Loading SimpleITK images and resizing them: "
        msg += str(num + 1) + " of " + str(len(filenames)) + "."
        print (msg, end = "\r")
        # Extract all images (3D)
        x = sitk.ReadImage(os.path.join(images_path, filename, 'image.nii.gz'))
        x = torch.from_numpy(sitk.GetArrayFromImage(x))
        try:
            x = centre_crop_pad_2d(x, img_size)
        except:
            print('Image could not be resized and will therefore be skipped: {}.'
            .format(filename))
            continue
        # Save new images so they can be loaded directly
        sitk.WriteImage(sitk.GetImageFromArray(x), 
            join_path([target_path, filename+".nii.gz"]))

def _extract_images_random(source_path, data_label, folder_name,
                           nr_images, nr_slices,
                           storage_data_path=storage_data_path):
    r"""Extracts MRI images and slices randomly based on input and saves
        the images."""
    images_path = source_path

    # Extract filenames
    filenames = set(os.listdir(images_path))

    # Define noise, in this case it is just a string contained in the filenames
    noise = 'KGU'
    # Select random images based on nr_images and random slices
    # for each image based on nr_slices
    random_data, image_names = select_random_images_slices(source_path, filenames, noise,
                                                           nr_images, nr_slices, nr_intensities=None)
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

    # Test scans
    labels['KGU-DC381269940F'+'_blur'] = 3/max_likert_value
    labels['KGU-DC381269940F'+'_downsample'] = 1/max_likert_value
    labels['KGU-DC381269940F'+'_ghosting'] = 1/max_likert_value
    labels['KGU-DC381269940F'+'_motion'] = 1/max_likert_value
    labels['KGU-DC381269940F'+'_noise'] = 1/max_likert_value
    labels['KGU-DC381269940F'+'_spike'] = 1/max_likert_value
    labels['KGU-DECED9094107'+'_blur'] = 2/max_likert_value
    labels['KGU-DECED9094107'+'_downsample'] = 1/max_likert_value
    labels['KGU-DECED9094107'+'_ghosting'] = 1/max_likert_value
    labels['KGU-DECED9094107'+'_motion'] = 1/max_likert_value
    labels['KGU-DECED9094107'+'_noise'] = 1/max_likert_value
    labels['KGU-DECED9094107'+'_spike'] = 1/max_likert_value
    labels['KGU-E9EC0F06F1D6'+'_blur'] = 1/max_likert_value
    labels['KGU-E9EC0F06F1D6'+'_downsample'] = 1/max_likert_value
    labels['KGU-E9EC0F06F1D6'+'_ghosting'] = 1/max_likert_value
    labels['KGU-E9EC0F06F1D6'+'_motion'] = 1/max_likert_value
    labels['KGU-E9EC0F06F1D6'+'_noise'] = 1/max_likert_value
    labels['KGU-E9EC0F06F1D6'+'_spike'] = 1/max_likert_value
    labels['KGU-E211D643E882'+'_blur'] = 1/max_likert_value
    labels['KGU-E211D643E882'+'_downsample'] = 1/max_likert_value
    labels['KGU-E211D643E882'+'_ghosting'] = 1/max_likert_value
    labels['KGU-E211D643E882'+'_motion'] = 1/max_likert_value
    labels['KGU-E211D643E882'+'_noise'] = 4/max_likert_value
    labels['KGU-E211D643E882'+'_spike'] = 1/max_likert_value
    labels['KGU-E2777160F425'+'_blur'] = 1/max_likert_value
    labels['KGU-E2777160F425'+'_downsample'] = 1/max_likert_value
    labels['KGU-E2777160F425'+'_ghosting'] = 1/max_likert_value
    labels['KGU-E2777160F425'+'_motion'] = 1/max_likert_value
    labels['KGU-E2777160F425'+'_noise'] = 2/max_likert_value
    labels['KGU-E2777160F425'+'_spike'] = 1/max_likert_value
    labels['KGU-EC922875308F'+'_blur'] = 2/max_likert_value
    labels['KGU-EC922875308F'+'_downsample'] = 1/max_likert_value
    labels['KGU-EC922875308F'+'_ghosting'] = 1/max_likert_value
    labels['KGU-EC922875308F'+'_motion'] = 1/max_likert_value
    labels['KGU-EC922875308F'+'_noise'] = 1/max_likert_value
    labels['KGU-EC922875308F'+'_spike'] = 1/max_likert_value
    
    # Train scans
    labels['KGU-1D1840AEB676'+'_blur'] = 2/max_likert_value
    labels['KGU-1D1840AEB676'+'_downsample'] = 1/max_likert_value
    labels['KGU-1D1840AEB676'+'_ghosting'] = 1/max_likert_value
    labels['KGU-1D1840AEB676'+'_motion'] = 1/max_likert_value
    labels['KGU-1D1840AEB676'+'_noise'] = 1/max_likert_value
    labels['KGU-1D1840AEB676'+'_spike'] = 1/max_likert_value
    labels['KGU-1DEA1FCA6643'+'_blur'] = 1/max_likert_value
    labels['KGU-1DEA1FCA6643'+'_downsample'] = 1/max_likert_value
    labels['KGU-1DEA1FCA6643'+'_ghosting'] = 1/max_likert_value
    labels['KGU-1DEA1FCA6643'+'_motion'] = 1/max_likert_value
    labels['KGU-1DEA1FCA6643'+'_noise'] = 1/max_likert_value
    labels['KGU-1DEA1FCA6643'+'_spike'] = 1/max_likert_value
    labels['KGU-2A20DFFE1EA9'+'_blur'] = 1/max_likert_value
    labels['KGU-2A20DFFE1EA9'+'_downsample'] = 1/max_likert_value
    labels['KGU-2A20DFFE1EA9'+'_ghosting'] = 1/max_likert_value
    labels['KGU-2A20DFFE1EA9'+'_motion'] = 1/max_likert_value
    labels['KGU-2A20DFFE1EA9'+'_noise'] = 3/max_likert_value
    labels['KGU-2A20DFFE1EA9'+'_spike'] = 1/max_likert_value
    labels['KGU-2B4799895867'+'_blur'] = 3/max_likert_value
    labels['KGU-2B4799895867'+'_downsample'] = 1/max_likert_value
    labels['KGU-2B4799895867'+'_ghosting'] = 1/max_likert_value
    labels['KGU-2B4799895867'+'_motion'] = 1/max_likert_value
    labels['KGU-2B4799895867'+'_noise'] = 1/max_likert_value
    labels['KGU-2B4799895867'+'_spike'] = 1/max_likert_value
    labels['KGU-04C14129E612'+'_blur'] = 1/max_likert_value
    labels['KGU-04C14129E612'+'_downsample'] = 1/max_likert_value
    labels['KGU-04C14129E612'+'_ghosting'] = 1/max_likert_value
    labels['KGU-04C14129E612'+'_motion'] = 1/max_likert_value
    labels['KGU-04C14129E612'+'_noise'] = 1/max_likert_value
    labels['KGU-04C14129E612'+'_spike'] = 1/max_likert_value
    labels['KGU-5D03B23D7168'+'_blur'] = 2/max_likert_value
    labels['KGU-5D03B23D7168'+'_downsample'] = 1/max_likert_value
    labels['KGU-5D03B23D7168'+'_ghosting'] = 1/max_likert_value
    labels['KGU-5D03B23D7168'+'_motion'] = 2/max_likert_value
    labels['KGU-5D03B23D7168'+'_noise'] = 2/max_likert_value
    labels['KGU-5D03B23D7168'+'_spike'] = 1/max_likert_value
    labels['KGU-8B22D8BA6ED2'+'_blur'] = 2/max_likert_value
    labels['KGU-8B22D8BA6ED2'+'_downsample'] = 1/max_likert_value
    labels['KGU-8B22D8BA6ED2'+'_ghosting'] = 1/max_likert_value
    labels['KGU-8B22D8BA6ED2'+'_motion'] = 2/max_likert_value
    labels['KGU-8B22D8BA6ED2'+'_noise'] = 2/max_likert_value
    labels['KGU-8B22D8BA6ED2'+'_spike'] = 1/max_likert_value
    labels['KGU-8DFCA3EE4A02'+'_blur'] = 3/max_likert_value
    labels['KGU-8DFCA3EE4A02'+'_downsample'] = 1/max_likert_value
    labels['KGU-8DFCA3EE4A02'+'_ghosting'] = 1/max_likert_value
    labels['KGU-8DFCA3EE4A02'+'_motion'] = 1/max_likert_value
    labels['KGU-8DFCA3EE4A02'+'_noise'] = 1/max_likert_value
    labels['KGU-8DFCA3EE4A02'+'_spike'] = 1/max_likert_value
    labels['KGU-9EB70F6120C5'+'_blur'] = 2/max_likert_value
    labels['KGU-9EB70F6120C5'+'_downsample'] = 1/max_likert_value
    labels['KGU-9EB70F6120C5'+'_ghosting'] = 1/max_likert_value
    labels['KGU-9EB70F6120C5'+'_motion'] = 3/max_likert_value
    labels['KGU-9EB70F6120C5'+'_noise'] = 1/max_likert_value
    labels['KGU-9EB70F6120C5'+'_spike'] = 1/max_likert_value
    labels['KGU-9FDEADEBE50D'+'_blur'] = 1/max_likert_value
    labels['KGU-9FDEADEBE50D'+'_downsample'] = 1/max_likert_value
    labels['KGU-9FDEADEBE50D'+'_ghosting'] = 1/max_likert_value
    labels['KGU-9FDEADEBE50D'+'_motion'] = 1/max_likert_value
    labels['KGU-9FDEADEBE50D'+'_noise'] = 1/max_likert_value
    labels['KGU-9FDEADEBE50D'+'_spike'] = 1/max_likert_value
    labels['KGU-14C83DA925D6'+'_blur'] = 3/max_likert_value
    labels['KGU-14C83DA925D6'+'_downsample'] = 1/max_likert_value
    labels['KGU-14C83DA925D6'+'_ghosting'] = 1/max_likert_value
    labels['KGU-14C83DA925D6'+'_motion'] = 1/max_likert_value
    labels['KGU-14C83DA925D6'+'_noise'] = 1/max_likert_value
    labels['KGU-14C83DA925D6'+'_spike'] = 1/max_likert_value
    labels['KGU-22A7B1A06992'+'_blur'] = 1/max_likert_value
    labels['KGU-22A7B1A06992'+'_downsample'] = 1/max_likert_value
    labels['KGU-22A7B1A06992'+'_ghosting'] = 1/max_likert_value
    labels['KGU-22A7B1A06992'+'_motion'] = 1/max_likert_value
    labels['KGU-22A7B1A06992'+'_noise'] = 2/max_likert_value
    labels['KGU-22A7B1A06992'+'_spike'] = 1/max_likert_value
    labels['KGU-28F1C7503A23'+'_blur'] = 1/max_likert_value
    labels['KGU-28F1C7503A23'+'_downsample'] = 1/max_likert_value
    labels['KGU-28F1C7503A23'+'_ghosting'] = 1/max_likert_value
    labels['KGU-28F1C7503A23'+'_motion'] = 1/max_likert_value
    labels['KGU-28F1C7503A23'+'_noise'] = 1/max_likert_value
    labels['KGU-28F1C7503A23'+'_spike'] = 1/max_likert_value
    labels['KGU-435E67EC1510'+'_blur'] = 1/max_likert_value
    labels['KGU-435E67EC1510'+'_downsample'] = 1/max_likert_value
    labels['KGU-435E67EC1510'+'_ghosting'] = 1/max_likert_value
    labels['KGU-435E67EC1510'+'_motion'] = 1/max_likert_value
    labels['KGU-435E67EC1510'+'_noise'] = 2/max_likert_value
    labels['KGU-435E67EC1510'+'_spike'] = 1/max_likert_value
    labels['KGU-675FF4A7E27A'+'_blur'] = 1/max_likert_value
    labels['KGU-675FF4A7E27A'+'_downsample'] = 1/max_likert_value
    labels['KGU-675FF4A7E27A'+'_ghosting'] = 1/max_likert_value
    labels['KGU-675FF4A7E27A'+'_motion'] = 1/max_likert_value
    labels['KGU-675FF4A7E27A'+'_noise'] = 2/max_likert_value
    labels['KGU-675FF4A7E27A'+'_spike'] = 1/max_likert_value
    labels['KGU-8160FACFB08D'+'_blur'] = 2/max_likert_value
    labels['KGU-8160FACFB08D'+'_downsample'] = 1/max_likert_value
    labels['KGU-8160FACFB08D'+'_ghosting'] = 1/max_likert_value
    labels['KGU-8160FACFB08D'+'_motion'] = 3/max_likert_value
    labels['KGU-8160FACFB08D'+'_noise'] = 1/max_likert_value
    labels['KGU-8160FACFB08D'+'_spike'] = 1/max_likert_value
    labels['KGU-9134A8F50ACB'+'_blur'] = 2/max_likert_value
    labels['KGU-9134A8F50ACB'+'_downsample'] = 1/max_likert_value
    labels['KGU-9134A8F50ACB'+'_ghosting'] = 1/max_likert_value
    labels['KGU-9134A8F50ACB'+'_motion'] = 2/max_likert_value
    labels['KGU-9134A8F50ACB'+'_noise'] = 1/max_likert_value
    labels['KGU-9134A8F50ACB'+'_spike'] = 1/max_likert_value
    labels['KGU-90011654B62A'+'_blur'] = 1/max_likert_value
    labels['KGU-90011654B62A'+'_downsample'] = 1/max_likert_value
    labels['KGU-90011654B62A'+'_ghosting'] = 1/max_likert_value
    labels['KGU-90011654B62A'+'_motion'] = 2/max_likert_value
    labels['KGU-90011654B62A'+'_noise'] = 1/max_likert_value
    labels['KGU-90011654B62A'+'_spike'] = 1/max_likert_value
    labels['KGU-361266990BF6'+'_blur'] = 1/max_likert_value
    labels['KGU-361266990BF6'+'_downsample'] = 1/max_likert_value
    labels['KGU-361266990BF6'+'_ghosting'] = 1/max_likert_value
    labels['KGU-361266990BF6'+'_motion'] = 1/max_likert_value
    labels['KGU-361266990BF6'+'_noise'] = 2/max_likert_value
    labels['KGU-361266990BF6'+'_spike'] = 1/max_likert_value
    labels['KGU-A9C48F0B68E1'+'_blur'] = 1/max_likert_value
    labels['KGU-A9C48F0B68E1'+'_downsample'] = 1/max_likert_value
    labels['KGU-A9C48F0B68E1'+'_ghosting'] = 1/max_likert_value
    labels['KGU-A9C48F0B68E1'+'_motion'] = 1/max_likert_value
    labels['KGU-A9C48F0B68E1'+'_noise'] = 2/max_likert_value
    labels['KGU-A9C48F0B68E1'+'_spike'] = 1/max_likert_value
    labels['KGU-AD3B5C1D2257'+'_blur'] = 3/max_likert_value
    labels['KGU-AD3B5C1D2257'+'_downsample'] = 1/max_likert_value
    labels['KGU-AD3B5C1D2257'+'_ghosting'] = 1/max_likert_value
    labels['KGU-AD3B5C1D2257'+'_motion'] = 1/max_likert_value
    labels['KGU-AD3B5C1D2257'+'_noise'] = 3/max_likert_value
    labels['KGU-AD3B5C1D2257'+'_spike'] = 1/max_likert_value
    labels['KGU-C2E218F3B192'+'_blur'] = 3/max_likert_value
    labels['KGU-C2E218F3B192'+'_downsample'] = 1/max_likert_value
    labels['KGU-C2E218F3B192'+'_ghosting'] = 1/max_likert_value
    labels['KGU-C2E218F3B192'+'_motion'] = 1/max_likert_value
    labels['KGU-C2E218F3B192'+'_noise'] = 1/max_likert_value
    labels['KGU-C2E218F3B192'+'_spike'] = 1/max_likert_value
    labels['KGU-C3F7B56742F2'+'_blur'] = 1/max_likert_value
    labels['KGU-C3F7B56742F2'+'_downsample'] = 1/max_likert_value
    labels['KGU-C3F7B56742F2'+'_ghosting'] = 1/max_likert_value
    labels['KGU-C3F7B56742F2'+'_motion'] = 1/max_likert_value
    labels['KGU-C3F7B56742F2'+'_noise'] = 2/max_likert_value
    labels['KGU-C3F7B56742F2'+'_spike'] = 1/max_likert_value
    labels['KGU-C0199AED68D5'+'_blur'] = 1/max_likert_value
    labels['KGU-C0199AED68D5'+'_downsample'] = 1/max_likert_value
    labels['KGU-C0199AED68D5'+'_ghosting'] = 1/max_likert_value
    labels['KGU-C0199AED68D5'+'_motion'] = 1/max_likert_value
    labels['KGU-C0199AED68D5'+'_noise'] = 2/max_likert_value
    labels['KGU-C0199AED68D5'+'_spike'] = 1/max_likert_value
    
    
    # Save the labels
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    with open(os.path.join(target_path, 'labels.json'), 'w') as fp:
        json.dump(labels, fp, sort_keys=True, indent=4)