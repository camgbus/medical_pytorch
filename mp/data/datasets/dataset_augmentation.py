# ------------------------------------------------------------------------------
# Data augmentation. Only, transformations from the TorchIO library are 
# used (https://torchio.readthedocs.io/transforms/augmentation.html).
# ------------------------------------------------------------------------------

# Imports
import torchio as tio
import torch
from mp.data.data import Data
from mp.paths import storage_data_path
import os
import SimpleITK as sitk
from mp.utils.load_restore import join_path
import json


# Load Data function
def load_dataset_seg(data, label_included=False):
    r""" This function loads data in form of SimpleITK images and returns
    them in form of a list. If label_included is true, the segmentation,
    i.e. label images will also be returned. Parallel, the image names will be
    transmitted in form of a second list. It takes a Dataset for segmentation,
    since the label path will be extracted."""
    itk_images = list()
    image_names = list()
    for ds_name, ds in data.datasets.items():
        for idx in range(ds.size):
            # Transform string to path
            label_str_path = str(ds.instances[idx].y.path)
            # Get image path
            img_str_path = label_str_path.replace('_gt', '')
            img_path = os.path.normpath(img_str_path)
            # Load Image in form of SimpleITK
            image = sitk.ReadImage(img_path)
            itk_images.append(image)
            image_names.append(img_str_path.split('.nii')[0].split('/')[-1])
            # Check if label is also needed
            if label_included:
                # Load label in form of SimpleITK
                label_path = os.path.normpath(label_str_path)
                label = sitk.ReadImage(label_path)
                itk_images.append(label)
                image_names.append(label_str_path.split('.nii')[0].split('/')[-1])

    return itk_images, image_names

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
                 storage_data_path=storage_data_path):
    r""" This function saves data in form of SimpleITK images at the specified
    location and folder based on the image names."""
    # Set target path
    target_path = os.path.join(storage_data_path, 'Augmentation', data_label, folder_name)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    if os.listdir(target_path):
        assert "Desired path (folder) already contains files!"

    # Save images
    for idx, image in enumerate(itk_images):
        sitk.WriteImage(image, 
            join_path([target_path, image_names[idx]+".nii.gz"]))

# Perfom augmentation on dataset
def augment_data_in_four_intensities(data, dataset_name, is_list=False,
                                     label_included=False, storage_data_path=storage_data_path):
    r""" This function takes a dataset and creates augmented datasets with 4 different intensities:
        - Downsampling
        - Blurring
        - Ghosting
        - Motion
        - (Gaussian) Noise
        - Spike
        It saves the data based on transmitted information and returns the data in form of a list
        with its labels as a dictionary."""

    # 0. Initialize variables
    aug_data = list()
    labels = dict()

    # 1. Check if data has been generated already:
    print('Check if data {} has been generated and can be retrieved from {}.'.format(dataset_name,
                                                                   os.path.join(storage_data_path,
                                                                            dataset_name, 'augmented_data')))
    if os.path.isdir(os.path.join(storage_data_path, dataset_name, 'data'))\
        and os.listdir(dataset_path):
        # Check if labels are present
        if os.path.isdir(os.path.join(storage_data_path, dataset_name, 'labels'))\
        and os.listdir(dataset_path):
            # Load data
            (_, _, filenames) = os.walk(os.path.join(storage_data_path,
                                        dataset_name, 'data')).next()
            for num, filename in enumerate(filenames):
                msg = "Loading SimpleITK images: "
                msg += str(num + 1) + " of " + str(len(filenames)) + "."
                print (msg, end = "\r")
                aug_data.append(sitk.ReadImage(img_path))
            
            # Load labels
            (_, _, filenames) = os.walk(os.path.join(storage_data_path,
                                        dataset_name, 'labels')).next()
            with open(os.path.join(storage_data_path,
            dataset_name, 'labels', filenames[0]), 'r') as fp:
                labels = json.load(fp)
            return aug_data, labels
        else:
            print('The labels are missing, i.e. data needs to be generated.')

    else:
        print('The directories do not exist or are empty, i.e. data needs to be generated.')      

    # 2. Load data as SimpleITK images
    itk_images, image_names = load_dataset(data, is_list, label_included)

    # 3. Define image names based on augmentation method and intensity:
    updated_image_names = list()
    updated_image_names.append([name + 'downsample_2' for name in image_names])
    updated_image_names.append([name + 'downsample_3' for name in image_names])
    updated_image_names.append([name + 'downsample_4' for name in image_names])
    updated_image_names.append([name + 'downsample_5' for name in image_names])
    updated_image_names.append([name + 'blur_2' for name in image_names])
    updated_image_names.append([name + 'blur_3' for name in image_names])
    updated_image_names.append([name + 'blur_4' for name in image_names])
    updated_image_names.append([name + 'blur_5' for name in image_names])
    updated_image_names.append([name + 'ghosting_2' for name in image_names])
    updated_image_names.append([name + 'ghosting_3' for name in image_names])
    updated_image_names.append([name + 'ghosting_4' for name in image_names])
    updated_image_names.append([name + 'ghosting_5' for name in image_names])
    updated_image_names.append([name + 'motion_2' for name in image_names])
    updated_image_names.append([name + 'motion_3' for name in image_names])
    updated_image_names.append([name + 'motion_4' for name in image_names])
    updated_image_names.append([name + 'motion_5' for name in image_names])
    updated_image_names.append([name + 'noise_2' for name in image_names])
    updated_image_names.append([name + 'noise_3' for name in image_names])
    updated_image_names.append([name + 'noise_4' for name in image_names])
    updated_image_names.append([name + 'noise_5' for name in image_names])
    updated_image_names.append([name + 'spike_2' for name in image_names])
    updated_image_names.append([name + 'spike_3' for name in image_names])
    updated_image_names.append([name + 'spike_4' for name in image_names])
    updated_image_names.append([name + 'spike_5' for name in image_names])

    # 4. Define labels based on augmentation method and intensity:
    for name in image_names:
        labels[name + 'downsample_2'] = torch.tensor([2.])
        labels[name + 'downsample_3'] = torch.tensor([3.])
        labels[name + 'downsample_4'] = torch.tensor([4.])
        labels[name + 'downsample_5'] = torch.tensor([5.])
        labels[name + 'blur_2'] = torch.tensor([2.])
        labels[name + 'blur_3'] = torch.tensor([3.])
        labels[name + 'blur_4'] = torch.tensor([4.])
        labels[name + 'blur_5'] = torch.tensor([5.])
        labels[name + 'ghosting_2'] = torch.tensor([2.])
        labels[name + 'ghosting_3'] = torch.tensor([3.])
        labels[name + 'ghosting_4'] = torch.tensor([4.])
        labels[name + 'ghosting_5'] = torch.tensor([5.])
        labels[name + 'motion_2'] = torch.tensor([2.])
        labels[name + 'motion_3'] = torch.tensor([3.])
        labels[name + 'motion_4'] = torch.tensor([4.])
        labels[name + 'motion_5'] = torch.tensor([5.])
        labels[name + 'noise_2'] = torch.tensor([2.])
        labels[name + 'noise_3'] = torch.tensor([3.])
        labels[name + 'noise_4'] = torch.tensor([4.])
        labels[name + 'noise_5'] = torch.tensor([5.])
        labels[name + 'spike_2'] = torch.tensor([2.])
        labels[name + 'spike_3'] = torch.tensor([3.])
        labels[name + 'spike_4'] = torch.tensor([4.])
        labels[name + 'spike_5'] = torch.tensor([5.])

    # 5. Define augmentation methods
    downsample2 = random_downsample(seed=42)
    downsample3 = random_downsample(seed=52)
    downsample4 = random_downsample(seed=62)
    downsample5 = random_downsample(seed=72)
    blur2 = random_blur(seed=0)
    blur3 = random_blur(seed=10)
    blur4 = random_blur(seed=20)
    blur5 = random_blur(seed=30)
    ghosting2 = random_ghosting(intensity=1.5,
                                seed=42)
    ghosting3 = random_ghosting(intensity=2.5,
                                seed=62)
    ghosting4 = random_ghosting(intensity=3.5,
                                seed=72)
    ghosting5 = random_ghosting(intensity=4.5,
                                seed=82)
    motion2 = random_motion(num_transforms=6,
                            image_interpolation='nearest',
                            seed=42)
    motion3 = random_motion(num_transforms=7,
                            image_interpolation='nearest',
                            seed=52)
    motion4 = random_motion(num_transforms=8,
                            image_interpolation='nearest',
                            seed=62)
    motion5 = random_motion(num_transforms=9,
                            image_interpolation='nearest',
                            seed=72)
    noise2 = random_noise(std=0.5,
                          seed=42)
    noise3 = random_noise(std=0.75,
                          seed=52)
    noise4 = random_noise(std=1.5,
                          seed=62)
    noise5 = random_noise(std=1.25,
                          seed=72)
    spike2 = random_spike(seed=42)
    spike3 = random_spike(seed=52)
    spike4 = random_spike(seed=62)
    spike5 = random_spike(seed=72)


    # 6. Apply augmentation methods to extracted data
    for num, image in enumerate(itk_images):
        msg = "Transforming SimpleITK images: "
        msg += str(num + 1) + " of " + str(len(itk_images)) + "."
        print (msg, end = "\r")
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
    print("Saving augmented images..")
    save_dataset(aug_data,
                 updated_image_names,
                 dataset_name,
                 'augmented_data',
                 storage_data_path)

    # 8. Save label dict
    print("Saving labels file..")
    with open(os.path.join(storage_data_path, dataset_name, 'labels',
    'labels.json', 'w')) as fp:
        json.dump(labels, fp, sort_keys=True, indent=4)

    # 9. Return augmented and labeled data
    print("Augmentation done.")
    return aug_data, labels
    
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
