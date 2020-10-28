# 1. Import needed libraries
import torch
from mp.data.data import Data
from mp.data.datasets.ds_mr_heart_decathlon import DecathlonLeftAtrium
import os
import mp.data.datasets.dataset_augmentation as augmentation
import SimpleITK as sitk
from mp.utils.load_restore import join_path


# 2. Define configuration dict and target_path
config = {'device':'cuda:4', 'input_shape': (1, 256, 256),
          'resize': False
    }
device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))


# 3. Define data
data = Data()
data.add_dataset(DecathlonLeftAtrium())


# 4. Load data as SimpleITK images where labels do not need to be augmented
itk_images, image_names = augmentation.load_dataset(data,
                                                    label_included=False)
itk_images_labels, image_label_names = augmentation.load_dataset(data,
                                                    label_included=True)


# 5. Define augmentation methods
affine = augmentation.random_affine(seed=43)
downsample = augmentation.random_downsample(seed=42)
deformation = augmentation.random_elastic_deformation(max_displacement=(40, 20, 0),
                                                      seed=0)
flip = augmentation.random_flip(axes=['superior-inferior'],
                                flip_probability=1)
bias = augmentation.random_bias_field(coefficients=1,
                                      seed=0)
blur = augmentation.random_blur(seed=0)
gamma = augmentation.random_gamma(seed=42)
ghosting = augmentation.random_ghosting(intensity=1.5,
                                        seed=42)
motion = augmentation.random_motion(num_transforms=6,
                                    image_interpolation='nearest',
                                    seed=42)
noise = augmentation.random_noise(std=0.5,
                                  seed=42)
spike = augmentation.random_spike(seed=42)
swap = augmentation.random_swap(seed=42)


# 6. Define output lists
affined_images = list()
downsampled_images = list()
deformed_images = list()
flipped_images = list()
biased_images = list()
blurred_images = list()
gamma_images = list()
ghosted_images = list()
motioned_images = list()
noised_images = list()
spiked_images = list()
swapped_images = list()


# 7. Apply augmentation methods to extracted data
for image in itk_images:
    affined_images.append(affine(image))
    downsampled_images.append(downsample(image))
    deformed_images.append(deformation(image))
    biased_images.append(bias(image))
    blurred_images.append(blur(image))
    gamma_images.append(gamma(image))
    ghosted_images.append(ghosting(image))
    motioned_images.append(motion(image))
    noised_images.append(noise(image))
    spiked_images.append(spike(image))
    swapped_images.append(swap(image))
# When images are flipped, the segmentation masks also need to be flipped!
for image in itk_images_labels:
    flipped_images.append(flip(image))


# 8. Save new images so they can be loaded directly
augmentation.save_dataset(affined_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Affined Images')
augmentation.save_dataset(downsampled_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Downsampled Images')
augmentation.save_dataset(deformed_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Deformed Images')
augmentation.save_dataset(flipped_images,
                          image_label_names,
                          'DecathlonLeftAtrium', 
                          'Flipped Images')
augmentation.save_dataset(biased_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Biased Images')
augmentation.save_dataset(blurred_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Blurred Images')
augmentation.save_dataset(gamma_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Gamma Images')
augmentation.save_dataset(ghosted_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Ghosted Images')
augmentation.save_dataset(motioned_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Motioned Images')
augmentation.save_dataset(noised_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Noised Images')
augmentation.save_dataset(spiked_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Spiked Images')
augmentation.save_dataset(swapped_images,
                          image_names,
                          'DecathlonLeftAtrium',
                          'Swapped Images')


# 8. Save image slices so they can be compared to check if everything worked fine
"""augmentation.save_random_vis_slice(itk_images,
                                   affined_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Affined Images')
augmentation.save_random_vis_slice(itk_images,
                                   downsampled_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Downsampled Images')
augmentation.save_random_vis_slice(itk_images,
                                   deformed_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Deformed Images')
augmentation.save_random_vis_slice(itk_images_labels,
                                   flipped_images,
                                   image_label_names,
                                   'DecathlonLeftAtrium',
                                   'Flipped Images')
augmentation.save_random_vis_slice(itk_images,
                                   biased_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Biased Images')
augmentation.save_random_vis_slice(itk_images,
                                   blurred_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Blurred Images')
augmentation.save_random_vis_slice(itk_images,
                                   gamma_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Gamma Images')
augmentation.save_random_vis_slice(itk_images,
                                   ghosted_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Ghosted Images')
augmentation.save_random_vis_slice(itk_images,
                                   motioned_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Motioned Images')
augmentation.save_random_vis_slice(itk_images,
                                   noised_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Noised Images')
augmentation.save_random_vis_slice(itk_images,
                                   spiked_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Spiked Images')
augmentation.save_random_vis_slice(sitk_images,
                                   swapped_images,
                                   image_names,
                                   'DecathlonLeftAtrium',
                                   'Swapped Images') """