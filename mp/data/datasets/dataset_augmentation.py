# ------------------------------------------------------------------------------
# Data augmentation. Only, transformations from the TorchIO library are 
# used (https://torchio.readthedocs.io/transforms/augmentation.html).
# ------------------------------------------------------------------------------

# Imports
import torch
import torchio as tio
from mp.data.data import Data
from mp.paths import storage_data_path
import random
import numpy as np
import os
import SimpleITK as sitk
from mp.utils.load_restore import join_path
import json
import numpy as np


# Load Data function
def load_dataset(data, is_list=False, label_included=False):
    r""" This function loads data in form of SimpleITK images and returns
    them in form of a list. If label_included is true, the segmentation,
    i.e. label images will also be returned. Parallel, the image names will be
    transmitted in form of a second list."""
    itk_images = list()
    image_names = list()
    if is_list:
        for idx in range(len(data)):
            # Load Image in form of SimpleITK
            image = data[idx].x.as_sitk()
            itk_images.append(image)
            name = data[idx].name
            image_names.append(name)
    else:
        for ds_name, ds in data.datasets.items():
            for idx in range(ds.size):
                # Load Image in form of SimpleITK
                image = ds.instances[idx].x.as_sitk()
                itk_images.append(image)
                image_names.append(ds.instances[idx].name)
                # Check if label is also needed
                if label_included:
                    # Load label in form of SimpleITK
                    label = ds.instances[idx].y.as_sitk()
                    itk_images.append(label)
                    image_names.append(ds.instances[idx].name + '_gt')

    return itk_images, image_names

# Save Data function
def save_dataset(itk_images, image_names, data_label, folder_name,
                 storage_data_path=storage_data_path, simpleITK=True,
                 empty_dir=False, extend=False):
    r""" This function saves data in form of SimpleITK images at the specified
    location and folder based on the image names. If simpleITK is set to false,
    the images will be transformed into numpy arrays and saved in form of a .npy
    file. empty_dir indicates if the directory should be emptied --> only
    necessary for random images for training, since each training should have new
    images. If extend is true, the images will be added to the directory if it
    contains already other files."""
    # Set target path
    target_path = os.path.join(storage_data_path, data_label, folder_name)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    if empty_dir and os.listdir(target_path):
        # Delete files in directory
        for f in os.listdir(target_path):
            os.remove(os.path.join(target_path, f))
    if not extend and os.listdir(target_path):
        assert "Desired path (folder) already contains files!"

    # Save images
    if simpleITK:
        for idx, image in enumerate(itk_images):
            if isinstance(image, np.ndarray):
                image = sitk.GetImageFromArray(image)
            sitk.WriteImage(image, 
                join_path([target_path, str(image_names[idx])+".nii.gz"]))
    else:
        images_np = list()
        for image in itk_images:
            image_np = sitk.GetArrayFromImage(image)
            images_np.append(image_np)
        np.save(join_path([target_path, data_label+'-trans_data_without_names.npy']), np.array(images_np))
        np.save(join_path([target_path, data_label+'-names_for_trans_data.npy']), np.array(image_names))


# Select randomly slices of images 
def select_random_images_slices(path, filenames, noise, nr_images, nr_slices):
    r""" This function selects randomly based on the number of images and noise type images
         from a path and filenames. The length of the filenames represents the number of instances,
         i.e. 3D volumes. If the number of images is greater than the actual images, then all images
         will be considered. From the loaded images randomly slices will be stacked and returned."""
    # Select only filenames based on noise type
    filenames = [filename for filename in filenames if noise in filename]

    # Reset nr_images if needed
    nr_images = len(filenames) if nr_images > len(filenames) else nr_images

    # Select random filenames from list based on nr_images
    if nr_images != len(filenames):
        filenames = random.sample(filenames, nr_images)

    # Loop through selected filenames and load images with defined nr_slices
    aug_data = list()
    image_names = list()
    for num, filename in enumerate(filenames):
        msg = 'Loading random (augmented) images as SimpleITK and extracting slices: '
        msg += str(num + 1) + ' of ' + str(len(filenames)) + '.'
        print (msg, end = '\r')
        if not '.nii.gz' in filename:
            continue
        image = sitk.ReadImage(os.path.join(path, filename))
        image = sitk.GetArrayFromImage(image)
        # If slices are greater than all possible slices in the image,
        # add the whole image
        if image.shape[0] < nr_slices:
            aug_data.append(image)
            image_names.append(filename.split('/')[-1].split('.nii')[0])
            continue
        # Select random indexes based on image depth
        slice_idx = random.sample(range(image.shape[0]), nr_slices)
        slice_idx.sort()
        img = list()
        # Loop through sorted index list and save the slices in list
        for idx in slice_idx:
            img.append(image[idx,:,:])
        image = np.array(img)
        image = sitk.GetImageFromArray(image)
        aug_data.append(image)
        image_names.append(filename.split('/')[-1].split('.nii')[0])

    return aug_data, image_names


# Perfom augmentation on dataset
def augment_data_in_four_intensities(data, dataset_name, is_list=False,
                                     label_included=False, storage_data_path=storage_data_path,
                                     max_likert_value=1, random=False, noise='blur', nr_images=100,
                                     nr_slices=50):
    r""" This function takes a dataset and creates augmented datasets with 4 different intensities:
        - Downsampling
        - Blurring
        - Ghosting
        - Motion
        - (Gaussian) Noise
        - Spike
        It saves the data based on transmitted information and returns the data in form of a list
        with its labels as a dictionary.
        If random is true, random images based on nr_images and nr_slices will be selected according to
        the noise type. If the noise type is set to 'all', from all noise types, random images will be
        selected."""

    # 0. Initialize variables
    aug_data = list()
    image_names = list()
    labels = dict()
    folder_name = 'randomised_data_' + str(noise)

    # 1. Check if data has been generated already:
    print('Check if data {} has been generated and can be retrieved from {}.'.format(dataset_name,
                                                                   os.path.join(storage_data_path,
                                                                 dataset_name, 'augmented_data')))
    if os.path.isdir(os.path.join(storage_data_path, dataset_name, 'augmented_data'))\
        and os.listdir(os.path.join(storage_data_path, dataset_name, 'augmented_data')):
        # Check if labels are present
        if os.path.isdir(os.path.join(storage_data_path, dataset_name, 'labels'))\
        and os.listdir(os.path.join(storage_data_path, dataset_name, 'labels')):
            filenames = list(file_name for file_name in
            os.listdir(os.path.join(storage_data_path,
                                   dataset_name, 'augmented_data'))
                                   if '._' not in file_name)
            
            # Load data
            path = os.path.join(storage_data_path, dataset_name, 'augmented_data')
            if random:
                aug_data, image_names = select_random_images_slices(path, filenames, noise,
                nr_images, nr_slices)
                # Save random images so they can be loaded
                print("Saving random images and image slices as SimpleITK for training and testing..")
                save_dataset(aug_data,
                             image_names,
                             dataset_name,
                             folder_name,
                             storage_data_path,
                             simpleITK=True,
                             empty_dir=True)
            else:
                for num, filename in enumerate(filenames):
                    msg = 'Loading augmented images as SimpleITK: '
                    msg += str(num + 1) + ' of ' + str(len(filenames)) + '.'
                    print (msg, end = '\r')
                    if not '.nii.gz' in filename:
                        continue
                    image_names.append(filename.split('/')[-1].split('.nii')[0])

            # Load labels
            filenames = os.listdir(os.path.join(storage_data_path,
                                   dataset_name, 'labels'))
            with open(os.path.join(storage_data_path,
            dataset_name, 'labels', filenames[0]), 'r') as fp:
                labels = json.load(fp)

            # Transform label integers into torch.tensors
            for key, value in labels.items():
                labels[key] = torch.tensor([value])
            return labels, image_names

        else:
            print('The labels are missing, i.e. data needs to be generated.')

    else:
        print('The directories do not exist or are empty, i.e. data needs to be generated.')      

    # 2. Load data as SimpleITK images
    itk_images, image_names = load_dataset(data, is_list, label_included)

    # 3. Define image names based on augmentation method and intensity:
    updated_image_names = list()
    for name in image_names:
        updated_image_names.append(str(name) + 'downsample_2')
        updated_image_names.append(str(name) + 'downsample_3')
        updated_image_names.append(str(name) + 'downsample_4')
        updated_image_names.append(str(name) + 'downsample_5')
        updated_image_names.append(str(name) + 'blur_2')
        updated_image_names.append(str(name) + 'blur_3')
        updated_image_names.append(str(name) + 'blur_4')
        updated_image_names.append(str(name) + 'blur_5')
        updated_image_names.append(str(name) + 'ghosting_2')
        updated_image_names.append(str(name) + 'ghosting_3')
        updated_image_names.append(str(name) + 'ghosting_4')
        updated_image_names.append(str(name) + 'ghosting_5')
        updated_image_names.append(str(name) + 'motion_2')
        updated_image_names.append(str(name) + 'motion_3')
        updated_image_names.append(str(name) + 'motion_4')
        updated_image_names.append(str(name) + 'motion_5')
        updated_image_names.append(str(name) + 'noise_2')
        updated_image_names.append(str(name) + 'noise_3')
        updated_image_names.append(str(name) + 'noise_4')
        updated_image_names.append(str(name) + 'noise_5')
        updated_image_names.append(str(name) + 'spike_2')
        updated_image_names.append(str(name) + 'spike_3')
        updated_image_names.append(str(name) + 'spike_4')
        updated_image_names.append(str(name) + 'spike_5')

    # 4. Define labels based on augmentation method and intensity:
    for name in image_names:
        labels[str(name) + 'downsample_2'] = 2/max_likert_value
        labels[str(name) + 'downsample_3'] = 3/max_likert_value
        labels[str(name) + 'downsample_4'] = 4/max_likert_value
        labels[str(name) + 'downsample_5'] = 5/max_likert_value
        labels[str(name) + 'blur_2'] = 2/max_likert_value
        labels[str(name) + 'blur_3'] = 3/max_likert_value
        labels[str(name) + 'blur_4'] = 4/max_likert_value
        labels[str(name) + 'blur_5'] = 5/max_likert_value
        labels[str(name) + 'ghosting_2'] = 2/max_likert_value
        labels[str(name) + 'ghosting_3'] = 3/max_likert_value
        labels[str(name) + 'ghosting_4'] = 4/max_likert_value
        labels[str(name) + 'ghosting_5'] = 5/max_likert_value
        labels[str(name) + 'motion_2'] = 2/max_likert_value
        labels[str(name) + 'motion_3'] = 3/max_likert_value
        labels[str(name) + 'motion_4'] = 4/max_likert_value
        labels[str(name) + 'motion_5'] = 5/max_likert_value
        labels[str(name) + 'noise_2'] = 2/max_likert_value
        labels[str(name) + 'noise_3'] = 3/max_likert_value
        labels[str(name) + 'noise_4'] = 4/max_likert_value
        labels[str(name) + 'noise_5'] = 5/max_likert_value
        labels[str(name) + 'spike_2'] = 2/max_likert_value
        labels[str(name) + 'spike_3'] = 3/max_likert_value
        labels[str(name) + 'spike_4'] = 4/max_likert_value
        labels[str(name) + 'spike_5'] = 5/max_likert_value

    # 5. Define augmentation methods
    downsample2 = random_downsample(axes=(0, 1, 2),
                                    downsampling=4,
                                    seed=42)
    downsample3 = random_downsample(axes=(0, 1, 2),
                                    downsampling=6,
                                    seed=42)
    downsample4 = random_downsample(axes=(0, 1, 2),
                                    downsampling=8,
                                    seed=42)
    downsample5 = random_downsample(axes=(0, 1, 2),
                                    downsampling=10,
                                    seed=42)
    blur2 = random_blur(std=1, seed=42)
    blur3 = random_blur(std=2, seed=42)
    blur4 = random_blur(std=3, seed=42)
    blur5 = random_blur(std=4, seed=42)
    ghosting2 = random_ghosting(intensity=0.55,
                                seed=42)
    ghosting3 = random_ghosting(intensity=0.95,
                                seed=42)
    ghosting4 = random_ghosting(intensity=1.35,
                                seed=42)
    ghosting5 = random_ghosting(intensity=1.75,
                                seed=42)
    motion2 = random_motion(degrees=13, translation=20,
                            num_transforms=2,
                            image_interpolation='lanczos',
                            seed=42)
    motion3 = random_motion(degrees=16, translation=25,
                            num_transforms=4,
                            image_interpolation='lanczos',
                            seed=52)
    motion4 = random_motion(degrees=19, translation=30,
                            num_transforms=6,
                            image_interpolation='lanczos',
                            seed=62)
    motion5 = random_motion(degrees=22, translation=35,
                            num_transforms=8,
                            image_interpolation='lanczos',
                            seed=72)
    noise2 = random_noise(200,200)
    noise3 = random_noise(375,375)
    noise4 = random_noise(550,550)
    noise5 = random_noise(775,775)
    spike2 = random_spike(num_spikes=5, intensity=15,
                          seed=42)
    spike3 = random_spike(num_spikes=10, intensity=15,
                          seed=42)
    spike4 = random_spike(num_spikes=18, intensity=18,
                          seed=42)
    spike5 = random_spike(num_spikes=25, intensity=25,
                          seed=42)


    # 6. Apply augmentation methods to extracted data
    for num, image in enumerate(itk_images):
        msg = "Transforming and saving SimpleITK images: "
        msg += str(num + 1) + " of " + str(len(itk_images)) + "."
        print (msg, end = "\r")
        img_names = updated_image_names[24*num:24*(num+1)]
        aug_data = list()
        aug_data.append(downsample2(image))
        aug_data.append(downsample3(image))
        aug_data.append(downsample4(image))
        aug_data.append(downsample5(image))
        aug_data.append(blur2(image))
        aug_data.append(blur3(image))
        aug_data.append(blur4(image))
        aug_data.append(blur5(image))
        aug_data.append(ghosting2(image))
        aug_data.append(ghosting3(image))
        aug_data.append(ghosting4(image))
        aug_data.append(ghosting5(image))
        aug_data.append(motion2(image))
        aug_data.append(motion3(image))
        aug_data.append(motion4(image))
        aug_data.append(motion5(image))
        aug_data.append(noise2(image))
        aug_data.append(noise3(image))
        aug_data.append(noise4(image))
        aug_data.append(noise5(image))
        aug_data.append(spike2(image))
        aug_data.append(spike3(image))
        aug_data.append(spike4(image))
        aug_data.append(spike5(image))
        # 7. Save new images so they can be loaded directly
        save_dataset(aug_data,
                 img_names,
                 dataset_name,
                 'augmented_data',
                 storage_data_path,
                 simpleITK=True,
                 extend=True)

    """
    # 7. Save new images so they can be loaded directly
    print('Saving augmented images as SimpleITK..')
    save_dataset(aug_data,
                 updated_image_names,
                 dataset_name,
                 'augmented_data',
                 storage_data_path,
                 simpleITK=True)"""
                 
    # 8. Save label dict
    print("Saving labels file..")
    file_path = os.path.join(storage_data_path, dataset_name, 'labels')
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    with open(os.path.join(file_path, 'labels.json'), 'w') as fp:
        json.dump(labels, fp, sort_keys=True, indent=4)

    # 9. Return labeled data
    print("Augmentation done.")
    # Transform label integers into torch.tensors
    for key, value in labels.items():
        labels[key] = torch.tensor([value])

    if random:
        path = os.path.join(storage_data_path, dataset_name, 'augmented_data')
        aug_data, image_names = select_random_images_slices(path, updated_image_names,
                                                            noise, nr_images,
                                                            nr_slices)
        # Save random images so they can be loaded
        print("Saving random images and image slices as SimpleITK for training and testing..")
        save_dataset(aug_data,
                     image_names,
                     dataset_name,
                     folder_name,
                     storage_data_path,
                     simpleITK=True,
                     empty_dir=True)
        return labels, image_names
    return labels, updated_image_names
    
# Spatial Functions for data Augmentation
def random_affine(scales=(0.9, 1.1), degrees=10, translation=0, isotropic=False,
                  center='image', default_pad_value='otsu',
                  image_interpolation='linear', p=1, seed=None, keys=None):
    r"""Random affine transformation.
    - scales: Tuple (a,b) defining the scaling magnitude. For example, using
        scales=(0.5, 0.5) will zoom out the image, making the objects inside
        look twice as small while preserving the physical size and position of
        the image.
    - degrees: Tuple (a,b) defining the rotation range in degrees.
    - translation: Tuple (a,b) defining the translation in mm.
    - isotropic: If True, the scaling factor along all dimensions is the same.
    - center: If 'image', rotations and scaling will be performed around
        the image center. If 'origin', rotations and scaling will be
        performed around the origin in world coordinates.
    - default_pad_value: As the image is rotated, some values near the
        borders will be undefined. If 'minimum', the fill value will
        be the image minimum. If 'mean', the fill value is the mean of
        the border values. If 'otsu', the fill value is the mean of the
        values at the border that lie under an Otsu threshold. If it is
        a number, that value will be used.
    - image_interpolation: 'nearest' can be used for quick experimentation as
        it is very fast, but produces relatively poor results. 'linear',
        default in TorchIO, is usually a good compromise between image
        quality and speed to be used for data augmentation during training.
        Methods such as 'bspline' or 'lanczos' generate high-quality
        results, but are generally slower. They can be used to obtain
        optimal resampling results during offline data preprocessing.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    affine = tio.RandomAffine(scales, degrees, translation, isotropic,
                              center, default_pad_value, image_interpolation,
                              p, seed, keys)
    return affine

def random_downsample(axes=(0, 1, 2), downsampling=(1.5, 5), p=1,
                      seed=None, keys=None):
    r"""Downsample an image along an axis. This transform simulates an
    image that has been acquired using anisotropic spacing, using
    downsampling with nearest neighbor interpolation.
    - axes: Axis or tuple of axes along which the image will be downsampled.
    - downsampling: Downsampling factor m > 1.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    downsample = tio.RandomDownsample(axes, downsampling, p, seed, keys)
    return downsample

def random_elastic_deformation(num_control_points=7, max_displacement=7.5,
                               locked_borders=2, image_interpolation='linear',
                               p=1, seed=None, keys=None):
    r"""Apply dense random elastic deformation. A random displacement is
    assigned to a coarse grid of control points around and inside the image.
    The displacement at each voxel is interpolated from the coarse grid
    using cubic B-splines.
    - num_control_points: Number of control points along each dimension of
        the coarse grid. Smaller numbers generate smoother deformations.
        The minimum number of control points is 4 as this transform
        uses cubic B-splines to interpolate displacement.
    - max_displacement: Maximum displacement along each dimension at each
        control point. 
    - locked_borders: If 0, all displacement vectors are kept. If 1,
        displacement of control points at the border of the coarse grid
        will also be set to 0. If 2, displacement of control points
        at the border of the image will also be set to 0.
    - image_interpolation: 'nearest' can be used for quick experimentation as
        it is very fast, but produces relatively poor results. 'linear',
        default in TorchIO, is usually a good compromise between image
        quality and speed to be used for data augmentation during training.
        Methods such as 'bspline' or 'lanczos' generate high-quality
        results, but are generally slower. They can be used to obtain
        optimal resampling results during offline data preprocessing.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    deformation = tio.RandomElasticDeformation(num_control_points,
                                               max_displacement,
                                               locked_borders,
                                               image_interpolation,
                                               p, seed, keys)
    return deformation

def random_flip(axes=0, flip_probability=0.5, p=1, seed=None, keys=None):
    r"""Reverse the order of elements in an image along the given axes.
    - axes: Index or tuple of indices of the spatial dimensions along
        which the image might be flipped. If they are integers, they must
        be in (0, 1, 2). Anatomical labels may also be used, such as
        'Left', 'Right', 'Anterior', 'Posterior', 'Inferior', 'Superior',
        'Height' and 'Width', 'AP' (antero-posterior), 'lr' (lateral),
        'w' (width) or 'i' (inferior). Only the first letter of the
        string will be used. If the image is 2D, 'Height' and 'Width'
        may be used.
    - flip_probability: Probability that the image will be flipped. This is
        computed on a per-axis basis.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    flip = tio.RandomFlip(axes, flip_probability, p, seed, keys)
    return flip


# Intensity Functions for data Augmentation
def random_bias_field(coefficients=0.5, order=3, p=1, seed=None, keys=None):
    r"""Add random MRI bias field artifact.
    - coefficients: Magnitude n of polynomial coefficients.
    - order: Order of the basis polynomial functions.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    bias = tio.RandomBiasField(coefficients, order, p, seed, keys)
    return bias

def random_blur(std=(0, 2), p=1, seed=None, keys=None):
    r"""Blur an image using a random-sized Gaussian filter.
    - std: Tuple (a,b) to compute the standard deviations (𝜎1,𝜎2,𝜎3)
        of the Gaussian kernels used to blur the image along each axis.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    blur = tio.RandomBlur(std, p, seed, keys)
    return blur

def random_gamma(log_gamma=(-0.3, 0.3), p=1, seed=None, keys=None):
    r"""Change contrast of an image by raising its values to the power γ.
    - log_gamma: Tuple (a,b), negative and positive values for this argument
        perform gamma compression and expansion.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    gamma = tio.RandomGamma(log_gamma, p, seed, keys)
    return gamma

def random_ghosting(num_ghosts=(4, 10), axes=(0, 1, 2), intensity=(0.5, 1),
                    restore=0.02, p=1, seed=None, keys=None):
    r"""Add random MRI ghosting artifact.
    - num_ghosts: Number of ‘ghosts’ n in the image.
    - axes: Axis along which the ghosts will be created. If axes is a
        tuple, the axis will be randomly chosen from the passed values.
        Anatomical labels may also be used.
    - intensity: Positive number representing the artifact strength s
        with respect to the maximum of the k-space. If 0, the ghosts
        will not be visible.
    - restore: Number between 0 and 1 indicating how much of the k-space
        center should be restored after removing the planes that generate
        the artifact.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    ghosting = tio.RandomGhosting(num_ghosts, axes, intensity, restore,
                               p, seed, keys)
    return ghosting

def random_motion(degrees=10, translation=10, num_transforms=2,
                  image_interpolation='linear', p=1, seed=None, keys=None):
    r"""Add random MRI motion artifact.
    - degrees: Tuple (a, b) defining the rotation range in degrees of
        the simulated movements.
    - translation: Tuple (a,b) defining the translation in mm
        of the simulated movements.
    - num_transforms: Number of simulated movements. Larger values generate
        more distorted images.
    - image_interpolation: 'nearest' can be used for quick experimentation as
        it is very fast, but produces relatively poor results. 'linear',
        default in TorchIO, is usually a good compromise between image
        quality and speed to be used for data augmentation during training.
        Methods such as 'bspline' or 'lanczos' generate high-quality
        results, but are generally slower. They can be used to obtain
        optimal resampling results during offline data preprocessing.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    motion = tio.RandomMotion(degrees, translation, num_transforms,
                              image_interpolation, p, seed, keys)
    return motion

def random_noise(mean=0, std=(0, 0.25), p=1, seed=None, keys=None):
    r"""Add random Gaussian noise.
    - mean: Mean μ of the Gaussian distribution from which the noise is sampled.
    - std: Standard deviation σ of the Gaussian distribution from which
        the noise is sampled.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    noise = tio.RandomNoise(mean, std, p, seed, keys)
    return noise

def random_spike(num_spikes=1, intensity=(1, 3), p=1, seed=None, keys=None):
    r"""Add random MRI spike artifacts.
    - num_spikes: Number of spikes n present in k-space. Larger values generate
        more distorted images.
    - intensity: Ratio r between the spike intensity and the maximum of
        the spectrum. Larger values generate more distorted images.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    spike = tio.RandomSpike(num_spikes, intensity, p, seed, keys)
    return spike

def random_swap(patch_size=15, num_iterations=100, p=1, seed=None, keys=None):
    r"""Randomly swap patches within an image.
    - patch_size: Tuple of integers (w,h,d) to swap patches of size w×h×d.
    - num_iterations: Number of times that two patches will be swapped.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    swap = tio.RandomSwap(patch_size, num_iterations, p, seed, keys)
    return swap