# 1. Import needed libraries
import torch
from mp.data.data import Data
from mp.data.datasets.ds_mr_heart_decathlon import DecathlonLeftAtrium
from mp.data.datasets.ds_mr_prostate_decathlon import DecathlonProstateT2
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
itk_images, image_names = augmentation.load_dataset(data, label_included=False)
itk_images_labels, image_label_names = augmentation.load_dataset(data, label_included=True)


# 5. Define augmentation methods
blur = augmentation.random_blur()
flip = augmentation.random_flip(axes=['inferior-superior'], flip_probability=1)


# 6. Define output lists
blurred_images = list()
flipped_images = list()


# 7. Apply augmentation methods to extracted data
for image in itk_images:
    blurred_images.append(blur(image))
# When images are flipped, the segmentation masks also need to be flipped!
for image in itk_images_labels:
    flipped_images.append(flip(image))


# 8. Save new images so they can be loaded directly
augmentation.save_dataset(blurred_images, image_names,
                          'DecathlonLeftAtrium', 'Blurred Images')
augmentation.save_dataset(flipped_images, image_label_names,
                          'DecathlonLeftAtrium', 'Flipped Images')