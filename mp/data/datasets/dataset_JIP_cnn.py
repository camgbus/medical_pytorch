# ------------------------------------------------------------------------------
# Dataset provided by JIP Tool.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import shutil
import torch
import json
import torchio as tio
import traceback
import numpy as np
import SimpleITK as sitk
# Install lungmask from https://github.com/amrane99/lungmask using pip install git+https://github.com/amrane99/lungmask
from lungmask import mask
from mp.data.pytorch.transformation import centre_crop_pad_3d
from mp.data.datasets.dataset_cnn import CNNDataset, CNNInstance
from mp.data.datasets.dataset_augmentation import augment_image_in_four_intensities as _augment_image


import random

class JIPDataset(CNNDataset):
    r"""Class for the dataset provided by the JIP tool/workflow.
    """
    def __init__(self, subset=None, img_size=(1, 60, 299, 299), max_likert_value=5, data_type='all', augmentation=False, gpu=True, cuda=0, msg_bot=False,
                 nr_images=20, preprocess=False, build_dataset=False, dtype='train', noise='blur'):
        r"""Constructor"""
        assert subset is None, "No subsets for this dataset."
        assert len(img_size) == 4, "Image size needs to be 4D --> (batch_size, depth, height, width)."
        self.img_size = img_size
        self.max_likert_value = max_likert_value
        self.augmentation = augmentation
        self.gpu = gpu
        self.cuda = cuda
        self.msg_bot = msg_bot
        self.data_type = data_type
        self.global_name = 'JIP'
        self.nr_images = nr_images
        self.data_path = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"]) # Inference Data
        self.data_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"])
        self.train_path = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"]) # Train Data
        self.train_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_TRAIN_DIR"])

        if preprocess:
            return self.preprocess()

        if build_dataset:
            instances = self.buildDataset(dtype, noise)
            super().__init__(instances=instances, name=self.global_name, modality='CT')

    def preprocess(self):
        r"""This function preprocesses (and augments) the input data."""
        # Delete data in directory and preprocess data.
        try:
            if self.data_type == 'inference':
                _delete_images_and_labels(self.data_dataset_path)
                _extract_images(self.data_path, self.data_dataset_path, self.img_size, self.augmentation, self.gpu, self.cuda)
                _generate_labels(self.max_likert_value, self.data_dataset_path, self.data_dataset_path)
                return True, None
            if self.data_type == 'train':
                _delete_images_and_labels(self.train_dataset_path)
                _extract_images(self.train_path, self.train_dataset_path, self.img_size, self.augmentation, self.gpu, self.cuda)
                _generate_labels(self.max_likert_value, self.train_dataset_path, self.train_dataset_path)
                return True, None
            if self.data_type == 'all':
                _delete_images_and_labels(self.data_dataset_path)
                _extract_images(self.data_path, self.data_dataset_path, self.img_size, self.augmentation, self.gpu, self.cuda)
                _generate_labels(self.max_likert_value, self.data_dataset_path, self.data_dataset_path)
                _delete_images_and_labels(self.train_dataset_path)
                _extract_images(self.train_path, self.train_dataset_path, self.img_size, self.augmentation, self.gpu, self.cuda)
                _generate_labels(self.max_likert_value, self.train_dataset_path, self.train_dataset_path)
                return True, None
        except: # catch *all* exceptions
            e = traceback.format_exc()
            return False, e

    def buildDataset(self, d_type, noise):
        r"""This function builds a dataset from the preprocessed (and augmented) data based on the d_type,
            either for training or inference. The d_type is the same as self.data_type, however it can not be
            'all' in this case, since it is important to be able to distinguish to which type a scan belongs
            (train -- inference). Noise specifies which data will be included in the dataset --> only used
            for training."""
        # Extract all images, if not already done
        if d_type == 'train':
            if not os.path.isdir(self.train_dataset_path) or not os.listdir(self.train_dataset_path):
                print("Train data needs to be preprocessed..")
                self.data_type = d_type
                self.preprocess()
        else:
            if not os.path.isdir(self.data_dataset_path) or not os.listdir(self.data_dataset_path):
                print("Inference data needs to be preprocessed..")
                self.data_type = d_type
                self.preprocess()

        # Assert if d_type is 'all'
        assert d_type != 'all', "The dataset type can not be all, it needs to be either 'train' or 'inference'!"

        # Build dataset based on d_type
        if d_type == 'inference':
            # Foldernames are patient_id
            study_names = [x for x in os.listdir(self.data_dataset_path) if 'DS_Store' not in x]

            # Build instances, dataset without labels!
            instances = list()
            print('\n')
            for num, name in enumerate(study_names):
                msg = 'Creating dataset from images: '
                msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                print (msg, end = '\r')
                if 'Decathlon' not in name:
                    a_name = name + '_' + str(noise)
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.data_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = None,
                        name = name,
                        group_id = None
                        ))
                elif 'Decathlon' in name or str(noise) in name:
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.data_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = None,
                        name = name,
                        group_id = None
                        ))

        if d_type == 'train':
            # Foldernames are patient_id
            study_names = [x for x in os.listdir(self.train_dataset_path) if 'DS_Store' not in x]

            # --- Remove when using the right model --> only for dummy! --- #
            noise_names = [x for x in study_names if 'DecathlonLung' in x and noise in x]
            decathlon_names = [x for x in study_names if 'DecathlonLung' in x and not 'blur' in x\
                               and not 'downsample' in x and not 'ghosting' in x and not 'motion' in x\
                               and not 'noise' in x and not 'spike' in x]
            decathlon_names = random.sample(decathlon_names, self.nr_images)

            # Select intensities equally
            for i in range(2, 6):
                intensity_names = [x for x in noise_names if '_' + str(noise) + str(i) in x]
                decathlon_names.extend(random.sample(intensity_names, self.nr_images))
            study_names = decathlon_names
            # --- Remove when using the right model --> only for dummy! --- #

            # Load labels and build one hot vector
            with open(os.path.join(self.train_dataset_path, 'labels.json'), 'r') as fp:
                labels = json.load(fp)
            one_hot = torch.nn.functional.one_hot(torch.arange(0, self.max_likert_value), num_classes=self.max_likert_value)

            # Build instances
            instances = list()
            print('\n')
            for num, name in enumerate(study_names):
                msg = 'Creating dataset from images: '
                msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                print (msg, end = '\r')
                if 'Decathlon' not in name:
                    a_name = name + '_' + str(noise)
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[a_name] * self.max_likert_value) - 1],
                        name = name,
                        group_id = None
                        ))
                elif 'Decathlon' in name or str(noise) in name:
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[name] * self.max_likert_value) - 1],
                        name = name,
                        group_id = None
                        ))
        return instances

def _delete_images_and_labels(path):
    r"""This function deletes every nifti and json (labels) file in the path."""
    # Walk through path and delete all .nii files
    print('Walk trough directory \'{}\' and delete nifti files..'.format(path))
    for dname, dirs, files in os.walk(path):
        for num, fname in enumerate(files):
            msg = str(num + 1) + '_ of ' + str(len(files)) + '_ file(s).'
            print (msg, end = '\r')
            # Check if file is a nifti file and delete it
            if '.nii' in fname or '.json' in fname:
                fpath = os.path.dirname(dname)
                shutil.rmtree(fpath)

def _extract_images(source_path, target_path, img_size=(1, 60, 299, 299), augmentation=False, gpu=False, cuda=0):
    r"""Extracts CT images and saves the modified images."""
    # Foldernames are patient_id
    filenames = [x for x in os.listdir(source_path) if 'DS_Store' not in x]
    
    # Define resample object (each image will be resampled to voxelsize (1, 1, 3))
    resample = tio.Resample((1, 1, 3))

    for num, filename in enumerate(filenames):
        msg = "Loading SimpleITK images/labels and center cropping them: "
        msg += str(num + 1) + " of " + str(len(filenames)) + "."
        print (msg, end = "\r")
        # Check if whole lung is captured
        discard, start_slc, end_slc = _whole_lung_captured(os.path.join(source_path, filename, 'img', 'img.nii.gz'), gpu, cuda)
     
        if not discard:
            # Extract all images (3D)
            x = resample(sitk.ReadImage(os.path.join(source_path, filename, 'img', 'img.nii.gz'))[:,:,start_slc:end_slc])
            y = resample(sitk.ReadImage(os.path.join(source_path, filename, 'seg', '001.nii.gz'))[:,:,start_slc:end_slc])
            x = torch.from_numpy(sitk.GetArrayFromImage(x)).unsqueeze_(0)
            y = torch.from_numpy(sitk.GetArrayFromImage(y).astype(np.int16)).unsqueeze_(0)
            try:
                x = centre_crop_pad_3d(x, img_size)[0]
                y = centre_crop_pad_3d(y, img_size)[0]
                if augmentation and 'DecathlonLung' in filename:
                    xs = list()
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='blur'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='downsample'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='ghosting'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='motion'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='noise'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='spike'))
            except:
                print('Image could not be resized/resampled/augmented and will therefore be skipped: {}.'
                .format(filename))
                continue
            # Create directories
            os.makedirs(os.path.join(target_path, filename, 'img'))
            os.makedirs(os.path.join(target_path, filename, 'seg'))
            # Save new images so they can be loaded directly
            sitk.WriteImage(sitk.GetImageFromArray(x), 
                os.path.join(target_path, filename, 'img', "img.nii.gz"))
            sitk.WriteImage(sitk.GetImageFromArray(y), 
                os.path.join(target_path, filename, 'seg', "001.nii.gz"))
            if augmentation and 'DecathlonLung' in filename:
                augmented = ['blur', 'downsample', 'ghosting', 'motion', 'noise', 'spike']
                for a_idx, a_type in enumerate(augmented):
                    for idx, i in enumerate(range(2, 6)):
                        # Build new filename
                        a_filename = filename.split('.')[0] + '_' + a_type + str(i)
                        # Make directories
                        os.makedirs(os.path.join(target_path, a_filename, 'img'))
                        os.makedirs(os.path.join(target_path, a_filename, 'seg'))
                        # Save augmented image and original label
                        sitk.WriteImage(xs[a_idx+idx],
                            os.path.join(target_path, a_filename, 'img', "img.nii.gz"))
                        sitk.WriteImage(sitk.GetImageFromArray(y),
                            os.path.join(target_path, filename, 'seg', "001.nii.gz"))


def _extract_lung_segmentation(input_path, gpu, cuda):
    r"""This function returns the lung segmentation of a CT scan."""
    # Load ct scan and create segmentation
    input_image = sitk.ReadImage(input_path)

    # load alternative models
    # model = mask.get_model('unet','LTRCLobes')
    # segmentation = mask.apply(input_image, model)

    segmentation = mask.apply(image=input_image, gpu=gpu, cuda=cuda.split(':')[-1])  # default model is U-net(R231)
    return segmentation

def _whole_lung_captured(input_path, gpu=True, cuda=0):
    r"""This function checks based on a CT scan, if the whole lung is captured."""
    # Calculate lung segmentation
    scan_np = _extract_lung_segmentation(input_path, gpu, cuda)

    # Check if Lung is whole captured
    start_seg = True
    end_seg = True
    discard = False
    start_seg_idx = None
    end_seg_idx = None
    for idx, ct_slice in enumerate(scan_np):
        transp = np.transpose(np.nonzero(ct_slice))
        if len(transp) != 0 and idx > 0:
            start_seg = False
            start_seg_idx = idx
            break
        if len(transp) != 0 and idx == 0:
            discard = True
            break

    if not discard:
        reversed_scan = scan_np[::-1]
        for idx, ct_slice in enumerate(reversed_scan):
            transp = np.transpose(np.nonzero(ct_slice))
            if len(transp) != 0 and idx > 0:
                end_seg = False
                end_seg_idx = len(reversed_scan) - idx - 1 # to get 0-based
                break
            if len(transp) != 0 and idx == 0:
                discard = True
                break

    return discard, start_seg_idx, end_seg_idx

def _generate_labels(max_likert_value, source_path, target_path):
    r"""This function generates the labels.json file that is necessary for training."""
    # Foldernames are patient_id
    filenames = [x for x in os.listdir(source_path) if 'DS_Store' not in x and 'DecathlonLung' in x\
                 and not 'blur' in x and not 'downsample' in x and not 'ghosting' in x and not 'motion' in x\
                 and not 'noise' in x and not 'spike' in x]

    # Generate labels for Decathlon with augmentation
    labels = dict()
    for name in filenames:
        labels[str(name)] = 1/max_likert_value
        labels[str(name) + '_blur2'] = 2/max_likert_value
        labels[str(name) + '_blur3'] = 3/max_likert_value
        labels[str(name) + '_blur4'] = 4/max_likert_value
        labels[str(name) + '_blur5'] = 5/max_likert_value
        labels[str(name) + '_downsample2'] = 2/max_likert_value
        labels[str(name) + '_downsample3'] = 3/max_likert_value
        labels[str(name) + '_downsample4'] = 4/max_likert_value
        labels[str(name) + '_downsample5'] = 5/max_likert_value
        labels[str(name) + '_ghosting2'] = 2/max_likert_value
        labels[str(name) + '_ghosting3'] = 3/max_likert_value
        labels[str(name) + '_ghosting4'] = 4/max_likert_value
        labels[str(name) + '_ghosting5'] = 5/max_likert_value
        labels[str(name) + '_motion2'] = 2/max_likert_value
        labels[str(name) + '_motion3'] = 3/max_likert_value
        labels[str(name) + '_motion4'] = 4/max_likert_value
        labels[str(name) + '_motion5'] = 5/max_likert_value
        labels[str(name) + '_noise2'] = 2/max_likert_value
        labels[str(name) + '_noise3'] = 3/max_likert_value
        labels[str(name) + '_noise4'] = 4/max_likert_value
        labels[str(name) + '_noise5'] = 5/max_likert_value
        labels[str(name) + '_spike2'] = 2/max_likert_value
        labels[str(name) + '_spike3'] = 3/max_likert_value
        labels[str(name) + '_spike4'] = 4/max_likert_value
        labels[str(name) + '_spike5'] = 5/max_likert_value

    # Add GC labels (defined by hand --> do not delete) to labels dict
    labels['GC_Corona_volume-covid19-A-0003'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0003'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0003'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0003'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0003'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0003'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0011'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0011'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0011'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0011'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0011'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0011'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0013'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0013'+'_downsample'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0013'+'_ghosting'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0013'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0013'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0013'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0014'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0014'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0014'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0014'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0014'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0014'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0016'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0016'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0016'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0016'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0016'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0016'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0025'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0025'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0025'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0025'+'_motion'] = 5/max_likert_value
    labels['GC_Corona_volume-covid19-A-0025'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0025'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0031'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0031'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0031'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0031'+'_motion'] = 4/max_likert_value
    labels['GC_Corona_volume-covid19-A-0031'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0031'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0034'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0034'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0034'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0034'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0034'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0034'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0038'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0038'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0038'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0038'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0038'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0038'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0039'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0039'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0039'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0039'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0039'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0039'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0041'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0041'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0041'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0041'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0041'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0041'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0044'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0044'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0044'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0044'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0044'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0044'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0046'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0046'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0046'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0046'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0046'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0046'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0047_1'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0047_1'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0047_1'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0047_1'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0047_1'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0047_1'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0053'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0053'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0053'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0053'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0053'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0053'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0054'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0054'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0054'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0054'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0054'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0054'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0066'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0066'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0066'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0066'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0066'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0066'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0070'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0070'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0070'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0070'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0070'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0070'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0072'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0072'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0072'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0072'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0072'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0072'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0073'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0073'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0073'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0073'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0073'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0073'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0074_1'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0074_1'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0074_1'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0074_1'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0074_1'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0074_1'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0077'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0077'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0077'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0077'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0077'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0077'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0083'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0083'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0083'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0083'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0083'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0083'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0090'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0090'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0090'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0090'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0090'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0090'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0092'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0092'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0092'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0092'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0092'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0092'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0096'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0096'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0096'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0096'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0096'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0096'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0106'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0106'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0106'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0106'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0106'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0106'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0110'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0110'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0110'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0110'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0110'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0110'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0112'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0112'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0112'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0112'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0112'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0112'+'_spike'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0114'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0114'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0114'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0114'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0114'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0114'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0120'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0120'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0120'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0120'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0120'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0120'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0129'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0129'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0129'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0129'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0129'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0129'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0130'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0130'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0130'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0130'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0130'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0130'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0133'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0133'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0133'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0133'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0133'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0133'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0147'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0147'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0147'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0147'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0147'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0147'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0151'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0151'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0151'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0151'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0151'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0151'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0154'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0154'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0154'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0154'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0154'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0154'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0161'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0161'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0161'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0161'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0161'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0161'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0164'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0164'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0164'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0164'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0164'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0164'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0165'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0165'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0165'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0165'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0165'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0165'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0167_1'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0167_1'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0167_1'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0167_1'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0167_1'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0167_1'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0173'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0173'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0173'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0173'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0173'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0173'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0178'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0178'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0178'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0178'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0178'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0178'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0179'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0179'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0179'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0179'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0179'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0179'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0181'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0181'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0181'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0181'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0181'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0181'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0187'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0187'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0187'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0187'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0187'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0187'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0196'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0196'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0196'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0196'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0196'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0196'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0199'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0199'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0199'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0199'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0199'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0199'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0201'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0201'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0201'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0201'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0201'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0201'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0202'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0202'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0202'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0202'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0202'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0202'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0214'+'_blur'] = 4/max_likert_value
    labels['GC_Corona_volume-covid19-A-0214'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0214'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0214'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0214'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0214'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0215'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0215'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0215'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0215'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0215'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0215'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0228'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0228'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0228'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0228'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0228'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0228'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0233'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0233'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0233'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0233'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0233'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0233'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0236'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0236'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0236'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0236'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0236'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0236'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0237'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0237'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0237'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0237'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0237'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0237'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0239'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0239'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0239'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0239'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0239'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0239'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0240'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0240'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0240'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0240'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0240'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0240'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0246'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0246'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0246'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0246'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0246'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0246'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0247'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0247'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0247'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0247'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0247'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0247'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0251'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0251'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0251'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0251'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0251'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0251'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0252'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0252'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0252'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0252'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0252'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0252'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0255'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0255'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0255'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0255'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0255'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0255'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0256_1'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0256_1'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0256_1'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0256_1'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0256_1'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0256_1'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0263'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0263'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0263'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0263'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0263'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0263'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0264'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0264'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0264'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0264'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0264'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0264'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0267'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0267'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0267'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0267'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0267'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0267'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0270'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0270'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0270'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0270'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0270'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0270'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0282'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0282'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0282'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0282'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0282'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0282'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0285'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0285'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0285'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0285'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0285'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0285'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0288'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0288'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0288'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0288'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0288'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0288'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0295'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0295'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0295'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0295'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0295'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0295'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0296'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0296'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0296'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0296'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0296'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0296'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0299'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0299'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0299'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0299'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0299'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0299'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0301'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0301'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0301'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0301'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0301'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0301'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0307'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0307'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0307'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0307'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0307'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0307'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0313'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0313'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0313'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0313'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0313'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0313'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0314'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0314'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0314'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0314'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0314'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0314'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0315'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0315'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0315'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0315'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0315'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0315'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0316'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0316'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0316'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0316'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0316'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0316'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0319'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0319'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0319'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0319'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0319'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0319'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0320'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0320'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0320'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0320'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0320'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0320'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0323'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0323'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0323'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0323'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0323'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0323'+'_spike'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0329'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0329'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0329'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0329'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0329'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0329'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0331'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0331'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0331'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0331'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0331'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0331'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0332'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0332'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0332'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0332'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0332'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0332'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0338'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0338'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0338'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0338'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0338'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0338'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0339'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0339'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0339'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0339'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0339'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0339'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0342'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0342'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0342'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0342'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0342'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0342'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0347'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0347'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0347'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0347'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0347'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0347'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0351'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0351'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0351'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0351'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0351'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0351'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0354'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0354'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0354'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0354'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0354'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0354'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0355'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0355'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0355'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0355'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0355'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0355'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0360'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0360'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0360'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0360'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0360'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0360'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0361'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0361'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0361'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0361'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0361'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0361'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0366'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0366'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0366'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0366'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0366'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0366'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0372'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0372'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0372'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0372'+'_motion'] = 4/max_likert_value
    labels['GC_Corona_volume-covid19-A-0372'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0372'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0377'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0377'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0377'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0377'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0377'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0377'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0380'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0380'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0380'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0380'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0380'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0380'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0382'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0382'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0382'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0382'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0382'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0382'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0383'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0383'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0383'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0383'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0383'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0383'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0386'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0386'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0386'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0386'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0386'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0386'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0387'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0387'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0387'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0387'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0387'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0387'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0388'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0388'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0388'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0388'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0388'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0388'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0391'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0391'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0391'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0391'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0391'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0391'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0392'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0392'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0392'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0392'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0392'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0392'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0394'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0394'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0394'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0394'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0394'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0394'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0397'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0397'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0397'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0397'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0397'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0397'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0400'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0400'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0400'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0400'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0400'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0400'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0402'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0402'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0402'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0402'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0402'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0402'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0407'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0407'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0407'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0407'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0407'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0407'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0413'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0413'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0413'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0413'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0413'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0413'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0414'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0414'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0414'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0414'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0414'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0414'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0416'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0416'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0416'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0416'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0416'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0416'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0417'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0417'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0417'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0417'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0417'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0417'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0418'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0418'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0418'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0418'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0418'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0418'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0421'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0421'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0421'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0421'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0421'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0421'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0422'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0422'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0422'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0422'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0422'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0422'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0423'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0423'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0423'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0423'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0423'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0423'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0435'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0435'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0435'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0435'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0435'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0435'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0437'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0437'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0437'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0437'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0437'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0437'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0443'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0443'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0443'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0443'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0443'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0443'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0445'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0445'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0445'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0445'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0445'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0445'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0455'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0455'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0455'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0455'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0455'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0455'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0462'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0462'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0462'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0462'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0462'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0462'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0463'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0463'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0463'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0463'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0463'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0463'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0464'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0464'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0464'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0464'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0464'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0464'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0472'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0472'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0472'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0472'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0472'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0472'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0473'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0473'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0473'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0473'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0473'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0473'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0475'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0475'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0475'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0475'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0475'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0475'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0476'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0476'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0476'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0476'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0476'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0476'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0479'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0479'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0479'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0479'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0479'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0479'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0483'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0483'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0483'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0483'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0483'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0483'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0494'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0494'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0494'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0494'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0494'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0494'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0495'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0495'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0495'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0495'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0495'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0495'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0498'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0498'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0498'+'_ghosting'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0498'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0498'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0498'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0500'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0500'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0500'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0500'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0500'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0500'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0502'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0502'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0502'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0502'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0502'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0502'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0504'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0504'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0504'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0504'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0504'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0504'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0511'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0511'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0511'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0511'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0511'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0511'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0521'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0521'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0521'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0521'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0521'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0521'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0522'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0522'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0522'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0522'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0522'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0522'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0524'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0524'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0524'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0524'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0524'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0524'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0525'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0525'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0525'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0525'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0525'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0525'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0526'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0526'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0526'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0526'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0526'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0526'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0530'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0530'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0530'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0530'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0530'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0530'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0531'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0531'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0531'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0531'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0531'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0531'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0534'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0534'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0534'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0534'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0534'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0534'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0537'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0537'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0537'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0537'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0537'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0537'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0547'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0547'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0547'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0547'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0547'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0547'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0548'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0548'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0548'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0548'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0548'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0548'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0553'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0553'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0553'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0553'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0553'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0553'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0557'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0557'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0557'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0557'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0557'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0557'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0559'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0559'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0559'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0559'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0559'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0559'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0560'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0560'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0560'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0560'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0560'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0560'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0562'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0562'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0562'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0562'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0562'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0562'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0566'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0566'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0566'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0566'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0566'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0566'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0567'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0567'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0567'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0567'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0567'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0567'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0569'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0569'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0569'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0569'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0569'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0569'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0570'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0570'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0570'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0570'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0570'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0570'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0573'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0573'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0573'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0573'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0573'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0573'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0575'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0575'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0575'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0575'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0575'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0575'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0576'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0576'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0576'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0576'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0576'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0576'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0579'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0579'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0579'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0579'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0579'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0579'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0581'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0581'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0581'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0581'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0581'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0581'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0585'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0585'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0585'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0585'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0585'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0585'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0586'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0586'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0586'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0586'+'_motion'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0586'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0586'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0589'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0589'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0589'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0589'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0589'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0589'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0590'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0590'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0590'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0590'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0590'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0590'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0599'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0599'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0599'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0599'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0599'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0599'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0604'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0604'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0604'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0604'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0604'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0604'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0612'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0612'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0612'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0612'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0612'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0612'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0614'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0614'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0614'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0614'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0614'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0614'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0623'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0623'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0623'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0623'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0623'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0623'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0626'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0626'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0626'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0626'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0626'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0626'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0627'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0627'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0627'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0627'+'_motion'] = 4/max_likert_value
    labels['GC_Corona_volume-covid19-A-0627'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0627'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0629'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0629'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0629'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0629'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0629'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0629'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0635'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0635'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0635'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0635'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0635'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0635'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0636'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0636'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0636'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0636'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0636'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0636'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0638'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0638'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0638'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0638'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0638'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0638'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0643'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0643'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0643'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0643'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0643'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0643'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0648'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0648'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0648'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0648'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0648'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0648'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0652'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0652'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0652'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0652'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0652'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0652'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0656'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0656'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0656'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0656'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0656'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0656'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0657'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0657'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0657'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0657'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0657'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0657'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0658'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0658'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0658'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0658'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0658'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0658'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0659'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0659'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0659'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0659'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0659'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0659'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0660'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0660'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0660'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0660'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0660'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0660'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0665'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0665'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0665'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0665'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0665'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0665'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0666'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0666'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0666'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0666'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0666'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0666'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0669'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0669'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0669'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0669'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0669'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0669'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0670'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0670'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0670'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0670'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0670'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0670'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0678'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0678'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0678'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0678'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0678'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0678'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0685'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0685'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0685'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0685'+'_motion'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0685'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0685'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0686'+'_blur'] = 2/max_likert_value
    labels['GC_Corona_volume-covid19-A-0686'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0686'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0686'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0686'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0686'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0694'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0694'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0694'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0694'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0694'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0694'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_blur'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0600'+'_spike'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0698'+'_blur'] = 3/max_likert_value
    labels['GC_Corona_volume-covid19-A-0698'+'_downsample'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0698'+'_ghosting'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0698'+'_motion'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0698'+'_noise'] = 1/max_likert_value
    labels['GC_Corona_volume-covid19-A-0698'+'_spike'] = 1/max_likert_value

    # Add FRA_UK labels (defined by hand --> do not delete) to labels dict
    labels['FRACorona_KGU-DC381269940F'+'_blur'] = 3/max_likert_value
    labels['FRACorona_KGU-DC381269940F'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-DC381269940F'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-DC381269940F'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-DC381269940F'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-DC381269940F'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-DECED9094107'+'_blur'] = 2/max_likert_value
    labels['FRACorona_KGU-DECED9094107'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-DECED9094107'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-DECED9094107'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-DECED9094107'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-DECED9094107'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-E9EC0F06F1D6'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-E9EC0F06F1D6'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-E9EC0F06F1D6'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-E9EC0F06F1D6'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-E9EC0F06F1D6'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-E9EC0F06F1D6'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-E211D643E882'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-E211D643E882'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-E211D643E882'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-E211D643E882'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-E211D643E882'+'_noise'] = 4/max_likert_value
    labels['FRACorona_KGU-E211D643E882'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-E2777160F425'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-E2777160F425'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-E2777160F425'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-E2777160F425'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-E2777160F425'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-E2777160F425'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-EC922875308F'+'_blur'] = 2/max_likert_value
    labels['FRACorona_KGU-EC922875308F'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-EC922875308F'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-EC922875308F'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-EC922875308F'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-EC922875308F'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-1D1840AEB676'+'_blur'] = 2/max_likert_value
    labels['FRACorona_KGU-1D1840AEB676'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-1D1840AEB676'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-1D1840AEB676'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-1D1840AEB676'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-1D1840AEB676'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-1DEA1FCA6643'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-1DEA1FCA6643'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-1DEA1FCA6643'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-1DEA1FCA6643'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-1DEA1FCA6643'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-1DEA1FCA6643'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-2A20DFFE1EA9'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-2A20DFFE1EA9'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-2A20DFFE1EA9'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-2A20DFFE1EA9'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-2A20DFFE1EA9'+'_noise'] = 3/max_likert_value
    labels['FRACorona_KGU-2A20DFFE1EA9'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-2B4799895867'+'_blur'] = 3/max_likert_value
    labels['FRACorona_KGU-2B4799895867'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-2B4799895867'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-2B4799895867'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-2B4799895867'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-2B4799895867'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-04C14129E612'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-04C14129E612'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-04C14129E612'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-04C14129E612'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-04C14129E612'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-04C14129E612'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-5D03B23D7168'+'_blur'] = 2/max_likert_value
    labels['FRACorona_KGU-5D03B23D7168'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-5D03B23D7168'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-5D03B23D7168'+'_motion'] = 2/max_likert_value
    labels['FRACorona_KGU-5D03B23D7168'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-5D03B23D7168'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-8B22D8BA6ED2'+'_blur'] = 2/max_likert_value
    labels['FRACorona_KGU-8B22D8BA6ED2'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-8B22D8BA6ED2'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-8B22D8BA6ED2'+'_motion'] = 2/max_likert_value
    labels['FRACorona_KGU-8B22D8BA6ED2'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-8B22D8BA6ED2'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-8DFCA3EE4A02'+'_blur'] = 3/max_likert_value
    labels['FRACorona_KGU-8DFCA3EE4A02'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-8DFCA3EE4A02'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-8DFCA3EE4A02'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-8DFCA3EE4A02'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-8DFCA3EE4A02'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-9EB70F6120C5'+'_blur'] = 2/max_likert_value
    labels['FRACorona_KGU-9EB70F6120C5'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-9EB70F6120C5'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-9EB70F6120C5'+'_motion'] = 3/max_likert_value
    labels['FRACorona_KGU-9EB70F6120C5'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-9EB70F6120C5'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-9FDEADEBE50D'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-9FDEADEBE50D'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-9FDEADEBE50D'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-9FDEADEBE50D'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-9FDEADEBE50D'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-9FDEADEBE50D'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-14C83DA925D6'+'_blur'] = 3/max_likert_value
    labels['FRACorona_KGU-14C83DA925D6'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-14C83DA925D6'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-14C83DA925D6'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-14C83DA925D6'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-14C83DA925D6'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-22A7B1A06992'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-22A7B1A06992'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-22A7B1A06992'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-22A7B1A06992'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-22A7B1A06992'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-22A7B1A06992'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-28F1C7503A23'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-28F1C7503A23'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-28F1C7503A23'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-28F1C7503A23'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-28F1C7503A23'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-28F1C7503A23'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-435E67EC1510'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-435E67EC1510'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-435E67EC1510'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-435E67EC1510'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-435E67EC1510'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-435E67EC1510'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-675FF4A7E27A'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-675FF4A7E27A'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-675FF4A7E27A'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-675FF4A7E27A'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-675FF4A7E27A'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-675FF4A7E27A'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-8160FACFB08D'+'_blur'] = 2/max_likert_value
    labels['FRACorona_KGU-8160FACFB08D'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-8160FACFB08D'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-8160FACFB08D'+'_motion'] = 3/max_likert_value
    labels['FRACorona_KGU-8160FACFB08D'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-8160FACFB08D'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-9134A8F50ACB'+'_blur'] = 2/max_likert_value
    labels['FRACorona_KGU-9134A8F50ACB'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-9134A8F50ACB'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-9134A8F50ACB'+'_motion'] = 2/max_likert_value
    labels['FRACorona_KGU-9134A8F50ACB'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-9134A8F50ACB'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-90011654B62A'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-90011654B62A'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-90011654B62A'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-90011654B62A'+'_motion'] = 2/max_likert_value
    labels['FRACorona_KGU-90011654B62A'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-90011654B62A'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-361266990BF6'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-361266990BF6'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-361266990BF6'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-361266990BF6'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-361266990BF6'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-361266990BF6'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-A9C48F0B68E1'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-A9C48F0B68E1'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-A9C48F0B68E1'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-A9C48F0B68E1'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-A9C48F0B68E1'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-A9C48F0B68E1'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-AD3B5C1D2257'+'_blur'] = 3/max_likert_value
    labels['FRACorona_KGU-AD3B5C1D2257'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-AD3B5C1D2257'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-AD3B5C1D2257'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-AD3B5C1D2257'+'_noise'] = 3/max_likert_value
    labels['FRACorona_KGU-AD3B5C1D2257'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-C2E218F3B192'+'_blur'] = 3/max_likert_value
    labels['FRACorona_KGU-C2E218F3B192'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-C2E218F3B192'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-C2E218F3B192'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-C2E218F3B192'+'_noise'] = 1/max_likert_value
    labels['FRACorona_KGU-C2E218F3B192'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-C3F7B56742F2'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-C3F7B56742F2'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-C3F7B56742F2'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-C3F7B56742F2'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-C3F7B56742F2'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-C3F7B56742F2'+'_spike'] = 1/max_likert_value
    labels['FRACorona_KGU-C0199AED68D5'+'_blur'] = 1/max_likert_value
    labels['FRACorona_KGU-C0199AED68D5'+'_downsample'] = 1/max_likert_value
    labels['FRACorona_KGU-C0199AED68D5'+'_ghosting'] = 1/max_likert_value
    labels['FRACorona_KGU-C0199AED68D5'+'_motion'] = 1/max_likert_value
    labels['FRACorona_KGU-C0199AED68D5'+'_noise'] = 2/max_likert_value
    labels['FRACorona_KGU-C0199AED68D5'+'_spike'] = 1/max_likert_value

    # Save labels
    print("Saving generated labels..")
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    with open(os.path.join(target_path, 'labels.json'), 'w') as fp:
        json.dump(labels, fp, sort_keys=True, indent=4)