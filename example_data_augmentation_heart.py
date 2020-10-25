# 1. Import needed libraries

import torch
from mp.paths import storage_data_path
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

target_path = os.path.join(storage_data_path, 'DecathlonLeftAtrium')


# 3. Define data

data = Data()
data.add_dataset(DecathlonLeftAtrium())



# 4. Load data as SimpleITK images

itk_images = list()
for ds_name, ds in data.datasets.items():
    for idx in range(ds.size):
        # Transform string to path
        img_path = os.path.normpath(ds.instances[idx].y.path)
        # Load Image in form pf SimpleITK
        image = sitk.ReadImage(img_path)
        itk_images.append(image)


# 5. Define augmentation methods

blur = augmentation.random_blur()
flip = augmentation.random_flip(axes=['inferior-superior'], flip_probability=1)

# 6. Define output lists

blurred_images = list()
flipped_images = list()


# 7. Apply augmentation methods to dextracted data

for image in itk_images:
    blurred_images.append(blur(image))
    flipped_images.append(flip(image))


# 8. Save new images so they can be loaded directly
for idx, elem in enumerate(flipped_images):
    sitk.WriteImage(elem, 
        join_path([target_path, str(idx)+".nii.gz"]))