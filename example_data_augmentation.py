# 1. Import needed libraries
import torch
from mp.data.data import Data
from mp.data.datasets.gc_corona_cnn import GCCorona
import os
import mp.data.datasets.dataset_augmentation as augmentation
import SimpleITK as sitk
from mp.utils.load_restore import join_path
from mp.paths import storage_data_path
import json


# 2. Define configuration dict and target_path
config = {'device':'cuda:4', 'input_shape': (1, 299, 299),
          'resize': False
    }
device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))


# 3. Define data
data = Data()
data.add_dataset(GCCorona(augmented=False))


# 4. Load data as SimpleITK images where labels do not need to be augmented
itk_images, image_names = augmentation.load_dataset(data, is_list=False,
                                                   label_included=False)


# 5. Define augmentation methods
"""blur2 = augmentation.random_blur(std=1, seed=42)
blur3 = augmentation.random_blur(std=2, seed=42)
blur4 = augmentation.random_blur(std=3, seed=42)
blur5 = augmentation.random_blur(std=4, seed=42)
downsample2 = augmentation.random_downsample(axes=(0, 1, 2),
                                    downsampling=4,
                                    seed=42)
downsample3 = augmentation.random_downsample(axes=(0, 1, 2),
                                downsampling=6,
                                seed=42)
downsample4 = augmentation.random_downsample(axes=(0, 1, 2),
                                downsampling=8,
                                seed=42)
downsample5 = augmentation.random_downsample(axes=(0, 1, 2),
                                downsampling=10,
                                seed=42)
ghosting2 = augmentation.random_ghosting(intensity=0.55,
                            seed=42)
ghosting3 = augmentation.random_ghosting(intensity=0.95,
                            seed=42)
ghosting4 = augmentation.random_ghosting(intensity=1.35,
                            seed=42)
ghosting5 = augmentation.random_ghosting(intensity=1.75,
                            seed=42)"""
motion2 = augmentation.random_motion(degrees=13, translation=20,
                        num_transforms=2,
                        image_interpolation='lanczos',
                        seed=42)
motion3 = augmentation.random_motion(degrees=16, translation=25,
                        num_transforms=4,
                        image_interpolation='lanczos',
                        seed=52)
motion4 = augmentation.random_motion(degrees=19, translation=30,
                        num_transforms=6,
                        image_interpolation='lanczos',
                        seed=62)
motion5 = augmentation.random_motion(degrees=22, translation=35,
                        num_transforms=8,
                        image_interpolation='lanczos',
                        seed=72)
"""noise2 = augmentation.random_noise(200,200)
noise3 = augmentation.random_noise(375,375)
noise4 = augmentation.random_noise(550,550)
noise5 = augmentation.random_noise(775,775)
spike2 = augmentation.random_spike(num_spikes=5, intensity=15,
                        seed=42)
spike3 = augmentation.random_spike(num_spikes=10, intensity=15,
                        seed=42)
spike4 = augmentation.random_spike(num_spikes=18, intensity=18,
                        seed=42)
spike5 = augmentation.random_spike(num_spikes=25, intensity=25,
                        seed=42)"""


# 6. Define output lists
images_new = list()

# 7. Apply augmentation methods to extracted data
for image in itk_images:
    """images_new.append(blur2(image))
    images_new.append(blur3(image))
    images_new.append(blur4(image))
    images_new.append(blur5(image))
    images_new.append(downsample2(image))
    images_new.append(downsample3(image))
    images_new.append(downsample4(image))
    images_new.append(downsample5(image))
    images_new.append(ghosting2(image))
    images_new.append(ghosting3(image))
    images_new.append(ghosting4(image))
    images_new.append(ghosting5(image))"""
    images_new.append(motion2(image))
    images_new.append(motion3(image))
    images_new.append(motion4(image))
    images_new.append(motion5(image))
    """images_new.append(noise2(image))
    images_new.append(noise3(image))
    images_new.append(noise4(image))
    images_new.append(noise5(image))
    images_new.append(spike2(image))
    images_new.append(spike3(image))
    images_new.append(spike4(image))
    images_new.append(spike5(image))"""

updated_image_names = list()
for name in image_names:
    """updated_image_names.append(str(name) + 'blur_2')
    updated_image_names.append(str(name) + 'blur_3')
    updated_image_names.append(str(name) + 'blur_4')
    updated_image_names.append(str(name) + 'blur_5')
    updated_image_names.append(str(name) + 'downsample_2')
    updated_image_names.append(str(name) + 'downsample_3')
    updated_image_names.append(str(name) + 'downsample_4')
    updated_image_names.append(str(name) + 'downsample_5')
    updated_image_names.append(str(name) + 'ghosting_2')
    updated_image_names.append(str(name) + 'ghosting_3')
    updated_image_names.append(str(name) + 'ghosting_4')
    updated_image_names.append(str(name) + 'ghosting_5')"""
    updated_image_names.append(str(name) + 'motion_2')
    updated_image_names.append(str(name) + 'motion_3')
    updated_image_names.append(str(name) + 'motion_4')
    updated_image_names.append(str(name) + 'motion_5')
    """updated_image_names.append(str(name) + 'noise_2')
    updated_image_names.append(str(name) + 'noise_3')
    updated_image_names.append(str(name) + 'noise_4')
    updated_image_names.append(str(name) + 'noise_5')
    updated_image_names.append(str(name) + 'spike_2')
    updated_image_names.append(str(name) + 'spike_3')
    updated_image_names.append(str(name) + 'spike_4')
    updated_image_names.append(str(name) + 'spike_5')"""

# 8. Save new images so they can be loaded directly
augmentation.save_dataset(images_new,
                          updated_image_names,
                          'GC_CoronaAugmented', 
                          'augmented_data')
"""
# 9. Generate labels
max_likert_value = 5
labels = dict()

for name in image_names:
    labels[str(name) + 'blur_2'] = 2/max_likert_value
    labels[str(name) + 'blur_3'] = 3/max_likert_value
    labels[str(name) + 'blur_4'] = 4/max_likert_value
    labels[str(name) + 'blur_5'] = 5/max_likert_value
    labels[str(name) + 'downsample_2'] = 2/max_likert_value
    labels[str(name) + 'downsample_3'] = 3/max_likert_value
    labels[str(name) + 'downsample_4'] = 4/max_likert_value
    labels[str(name) + 'downsample_5'] = 5/max_likert_value
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

file_path = os.path.join(storage_data_path, 'GC_ChallengeAugmented', 'labels')
if not os.path.isdir(file_path):
    os.makedirs(file_path)

with open(os.path.join(file_path, 'labels.json'), 'w') as fp:
    json.dump(labels, fp, sort_keys=True, indent=4)"""

print("Augmentation done")