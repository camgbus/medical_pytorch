# ------------------------------------------------------------------------------
# Dataset provided by JIP Tool.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import shutil
import torch
import json
import random
import torchio as tio
import traceback
import numpy as np
import SimpleITK as sitk
from mp.data.pytorch.transformation import centre_crop_pad_3d
from mp.data.datasets.dataset_cnn import CNNDataset, CNNInstance
from mp.data.datasets.dataset_augmentation import augment_image_in_four_intensities as _augment_image
from mp.utils.lung_captured import whole_lung_captured as LungFullyCaptured
from mp.utils.generate_labels import generate_train_labels, generate_test_labels
from mp.data.datasets.dataset_utils import delete_images_and_labels

class JIPDataset(CNNDataset):
    r"""Class for the dataset provided by the JIP tool/workflow.
    """
    def __init__(self, subset=None, img_size=(1, 60, 299, 299), num_intensities=5, data_type='all', augmentation=False, sample_size=25, gpu=True, cuda=0, msg_bot=False,
                 nr_images=20, build_dataset=False, dtype='train', noise='blur', ds_name='Decathlon', seed=42):
        r"""Constructor"""
        assert subset is None, "No subsets for this dataset."
        assert len(img_size) == 4, "Image size needs to be 4D --> (batch_size, depth, height, width)."
        self.img_size = img_size
        self.num_intensities = num_intensities
        self.augmentation = augmentation
        self.sample_size = sample_size
        self.gpu = gpu
        self.cuda = cuda
        self.msg_bot = msg_bot
        self.data_type = data_type
        self.ds_name = ds_name
        self.nr_images = nr_images
        self.data_path = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"]) # Inference Data
        self.data_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"])
        self.train_path = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"]) # Train Data
        self.train_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_TRAIN_DIR"])
        self.test_path = os.path.join(os.environ["TEST_WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"]) # Test Data
        self.test_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_TEST_DIR"])

        if build_dataset:
            instances = self.buildDataset(dtype, noise, seed)
            super().__init__(instances=instances, name=self.ds_name, modality='CT')

    def preprocess(self):
        r"""This function preprocesses (and augments) the input data."""
        # Delete data in directory and preprocess data.
        try:
            if self.data_type == 'inference':
                delete_images_and_labels(self.data_dataset_path)
                _extract_images(self.data_path, self.data_dataset_path, self.img_size, False, 0, self.gpu, self.cuda, True)
                return True, None
            if self.data_type == 'train':
                delete_images_and_labels(self.train_dataset_path)
                _extract_images(self.train_path, self.train_dataset_path, self.img_size, self.augmentation, self.sample_size, self.gpu, self.cuda)
                generate_train_labels(self.num_intensities, self.train_dataset_path, self.train_dataset_path)
                return True, None
            if self.data_type == 'test':
                delete_images_and_labels(self.test_dataset_path)
                _extract_images(self.test_path, self.test_dataset_path, self.img_size, False, 0, self.gpu, self.cuda, True)
                generate_test_labels(self.num_intensities, self.test_dataset_path, self.test_dataset_path)
                return True, None
            if self.data_type == 'all':
                delete_images_and_labels(self.data_dataset_path)
                _extract_images(self.data_path, self.data_dataset_path, self.img_size, False, 0, self.gpu, self.cuda, True)
                delete_images_and_labels(self.train_dataset_path)
                _extract_images(self.train_path, self.train_dataset_path, self.img_size, self.augmentation, self.sample_size, self.gpu, self.cuda)
                generate_train_labels(self.num_intensities, self.train_dataset_path, self.train_dataset_path)
                delete_images_and_labels(self.test_dataset_path)
                _extract_images(self.test_path, self.test_dataset_path, self.img_size, False, 0, self.gpu, self.cuda, True)
                generate_test_labels(self.num_intensities, self.test_dataset_path, self.test_dataset_path)
                return True, None
        except: # catch *all* exceptions
            e = traceback.format_exc()
            return False, e

    def buildDataset(self, d_type, noise, seed):
        r"""This function builds a dataset from the preprocessed (and augmented) data based on the d_type,
            either for training or inference. The d_type is the same as self.data_type, however it can not be
            'all' in this case, since it is important to be able to distinguish to which type a scan belongs
            (train -- inference). Noise specifies which data will be included in the dataset --> only used
            for training. ds_name specifies which dataset should be build, based on its name (in foldername).
            This can be 'Decathlon', 'GC' or 'FRA'. ds_name is only necessary for d_type == 'train'."""
        # Extract all images, if not already done
        if d_type == 'train':
            if not os.path.isdir(self.train_dataset_path) or not os.listdir(self.train_dataset_path):
                print("Train data needs to be preprocessed..")
                self.data_type = d_type
                self.preprocess()
        elif d_type == 'test':
            if not os.path.isdir(self.test_dataset_path) or not os.listdir(self.test_dataset_path):
                print("Test data needs to be preprocessed..")
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
            study_names = [x for x in os.listdir(self.data_dataset_path) if 'DS_Store' not in x and '._' not in x]

            # Build instances, dataset without labels!
            instances = list()
            print()
            for num, name in enumerate(study_names):
                msg = 'Creating inference dataset from images: '
                msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                print (msg, end = '\r')
                instances.append(CNNInstance(
                    x_path = os.path.join(self.data_dataset_path, name, 'img', 'img.nii.gz'),
                    y_label = None,
                    name = name,
                    group_id = None
                    ))

        if d_type == 'train':
            # Foldernames are patient_id
            study_names = [x for x in os.listdir(self.train_dataset_path) if 'DS_Store' not in x and '._' not in x]

            # Load labels and build one hot vector
            with open(os.path.join(self.train_dataset_path, 'labels.json'), 'r') as fp:
                labels = json.load(fp)
            one_hot = torch.nn.functional.one_hot(torch.arange(0, self.num_intensities), num_classes=self.num_intensities)

            # Build instances list
            instances = list()
            print()

            if self.ds_name == 'Decathlon':
                study_names = _get_equally_distributed_names(study_names, self.ds_name, noise, self.nr_images, self.num_intensities, seed)
                # Build instances
                for num, name in enumerate(study_names):
                    msg = 'Creating dataset from images: '
                    msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                    print (msg, end = '\r')
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[name] * self.num_intensities) - 1],
                        name = name,
                        group_id = None
                        ))

            elif self.ds_name == 'GC':
                study_names = _get_equally_distributed_names(study_names, self.ds_name, noise, self.nr_images, self.num_intensities, seed)
                """
                GC_names = [x for x in study_names if self.ds_name in x]
                if len(GC_names) > 5 * self.nr_images:
                    GC_names = random.sample(GC_names, 5 * self.nr_images)
                study_names = GC_names"""
                # Build instances
                for num, name in enumerate(study_names):
                    msg = 'Creating dataset from images: '
                    msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                    print (msg, end = '\r')
                    a_name = name + '_' + str(noise)
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[a_name] * self.num_intensities) - 1],
                        name = name,
                        group_id = None
                        ))

            elif self.ds_name == 'FRA':
                study_names = _get_equally_distributed_names(study_names, self.ds_name, noise, self.nr_images, self.num_intensities, seed)
                """
                FRA_names = [x for x in study_names if self.ds_name in x]
                if len(FRA_names) > 5 * self.nr_images:
                    FRA_names = random.sample(FRA_names, 5 * self.nr_images)
                study_names = FRA_names"""
                # Build instances
                for num, name in enumerate(study_names):
                    msg = 'Creating dataset from images: '
                    msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                    print (msg, end = '\r')
                    a_name = name + '_' + str(noise)
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[a_name] * self.num_intensities) - 1],
                        name = name,
                        group_id = None
                        ))

            elif self.ds_name == 'mixed':
                # Decathlon + GC + FRA
                Decathlon_names = _get_equally_distributed_names(study_names, 'Decathlon', noise, self.nr_images, self.num_intensities, seed)
                GC_names = _get_equally_distributed_names(study_names, 'GC', noise, self.nr_images, self.num_intensities, seed)
                FRA_names = _get_equally_distributed_names(study_names, 'FRA', noise, self.nr_images, self.num_intensities, seed)
                """
                # Decathlon + 2 x self.nr_images random GC and 2 x self.nr_images random FRA
                Decathlon_names = _get_equally_distributed_names(study_names, 'Decathlon', noise, self.nr_images, self.num_intensities, seed)
                GC_names = [x for x in study_names if 'GC' in x]
                FRA_names = [x for x in study_names if 'FRA' in x]
                if len(GC_names) > 2 * self.nr_images:
                    GC_names = random.sample(GC_names, 2 * self.nr_images)
                if len(FRA_names) > 2 * self.nr_images:
                    FRA_names = random.sample(FRA_names, 2 * self.nr_images)"""
                study_names = Decathlon_names + GC_names + FRA_names
                # Build instances for Decathlon, GC and FRA
                for num, name in enumerate(study_names):
                    msg = 'Creating dataset from images: '
                    msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                    print (msg, end = '\r')
                    if 'Decathlon' in name:
                        a_name = name
                    else:
                        a_name = name + '_' + str(noise)
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[a_name] * self.num_intensities) - 1],
                        name = name,
                        group_id = None
                        ))

            else:
                # Retraining performed by institutes with their own dataset.
                # NOTE: Data is already preprocessed and saved under
                # preprocessed_dirs/output_train in defined structure. --> All this data will be loaded!
                # NOTE: Labels dictionary for this data is saved under preprocessed_dirs/output_train/labels.json
                # Build instances
                for num, name in enumerate(study_names):
                    msg = 'Creating dataset from images: '
                    msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                    print (msg, end = '\r')
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[name] * self.num_intensities) - 1],
                        name = name,
                        group_id = None
                        ))

        if d_type == 'test':
            # Foldernames are patient_id
            study_names = [x for x in os.listdir(self.test_dataset_path) if 'DS_Store' not in x and '._' not in x]
            
            # Load labels and build one hot vector
            with open(os.path.join(self.test_dataset_path, 'labels.json'), 'r') as fp:
                labels = json.load(fp)
            one_hot = torch.nn.functional.one_hot(torch.arange(0, self.num_intensities), num_classes=self.num_intensities)

            # Build instances, dataset without labels!
            instances = list()
            print()

            # Build instances
            for num, name in enumerate(study_names):
                msg = 'Creating test dataset from images: '
                msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                print (msg, end = '\r')
                instances.append(CNNInstance(
                    x_path = os.path.join(self.test_dataset_path, name, 'img', 'img.nii.gz'),
                    y_label = one_hot[int(labels[name] * self.num_intensities) - 1],
                    name = name,
                    group_id = None
                    ))

        return instances


def _get_equally_distributed_names(study_names, ds_name, noise, nr_images, num_intensities, seed):
    r"""Extracts a list of folder names representing ds_name Dataset, based on noise and nr_images.
        An equal distribution of images will be extracted, ie. nr_images from each intensity level resulting
        in a dataset of num_intensities x nr_images foldernames."""
    # Set random seed
    random.seed(seed)

    # Extract filenames
    noise_names = [x for x in study_names if ds_name in x and noise in x] # Augmented scans
    ds_names = [x for x in study_names if ds_name in x and not 'blur' in x\
                    and not 'resolution' in x and not 'ghosting' in x and not 'motion' in x\
                    and not 'noise' in x and not 'spike' in x] # Original scans
    if len(ds_names) > nr_images:
        ds_names = random.sample(ds_names, nr_images)

    # Select intensities equally
    for i in range(1, num_intensities):
        intensity_names = [x for x in noise_names if '_' + str(noise) + str(i) in x]
        if len(intensity_names) > nr_images:
            ds_names.extend(random.sample(intensity_names, nr_images))
        else:
            ds_names.extend(intensity_names)
    # Reset random seed
    random.seed()
    return ds_names


def _extract_images(source_path, target_path, img_size=(1, 60, 299, 299), augmentation=False, sample_size=25, gpu=False, cuda=0, no_use_lfc=False):
    r"""Extracts CT images and saves the modified images. Augmentation will be performed on Decathlon, FRA and GC data if desired.
        Only sample_size images will be augmented, not all of them!"""
    # Foldernames are patient_id
    filenames = [x for x in os.listdir(source_path) if 'DS_Store' not in x and '._' not in x]

    # Random foldernames that will be augmented if desired
    Decathlon = [x for x in filenames if 'Decathlon' in x]
    FRA = [x for x in filenames if 'FRA' in x]
    GC = [x for x in filenames if 'GC' in x]
    # Extract samples
    if len(Decathlon) > sample_size:
        Decathlon = random.sample(Decathlon, sample_size)
    if len(FRA) > sample_size:
        FRA = random.sample(FRA, sample_size)
    if len(GC) > sample_size:
        GC = random.sample(GC, sample_size)
    files_to_aug = Decathlon + FRA + GC
    
    # Define resample object (each image will be resampled to voxelsize (1, 1, 3))
    resample = tio.Resample((1, 1, 3))

    for num, filename in enumerate(filenames):
        msg = "Loading SimpleITK images/labels and center cropping them: "
        msg += str(num + 1) + " of " + str(len(filenames)) + "."
        print (msg, end = "\r")

        # Set default value for discard, in case for inference, where the LungFullyCaptured is not used!
        discard = False
        start_slc = 0
        end_slc = -1
        if not no_use_lfc:
            # Check if whole lung is captured
            discard, start_slc, end_slc = LungFullyCaptured(os.path.join(source_path, filename, 'img', 'img.nii.gz'), gpu, cuda)
     
        if not discard:
            # Extract all images (3D)
            x = resample(sitk.ReadImage(os.path.join(source_path, filename, 'img', 'img.nii.gz'))[:,:,start_slc:end_slc])
            x = torch.from_numpy(sitk.GetArrayFromImage(x)).unsqueeze_(0)
            if not no_use_lfc:
                y = resample(sitk.ReadImage(os.path.join(source_path, filename, 'seg', '001.nii.gz'))[:,:,start_slc:end_slc])
                y = torch.from_numpy(sitk.GetArrayFromImage(y).astype(np.int16)).unsqueeze_(0)
            try:
                x = centre_crop_pad_3d(x, img_size)[0]
                if not no_use_lfc:
                    y = centre_crop_pad_3d(y, img_size)[0]
                if augmentation and filename in files_to_aug: # Do augmentation
                    xs = list()
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='blur'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='resolution'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='ghosting'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='motion'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='noise'))
                    xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='spike'))
            except:
                print('Image could not be resized/resampled/augmented and will therefore be skipped: {}.'
                .format(filename))
                continue
            # Create directories
            # Save new images so they can be loaded directly
            os.makedirs(os.path.join(target_path, filename, 'img'))
            sitk.WriteImage(sitk.GetImageFromArray(x), 
                os.path.join(target_path, filename, 'img', "img.nii.gz"))
            if not no_use_lfc:
                os.makedirs(os.path.join(target_path, filename, 'seg'))
                sitk.WriteImage(sitk.GetImageFromArray(y), 
                    os.path.join(target_path, filename, 'seg', "001.nii.gz"))
            if augmentation and filename in files_to_aug:
                augmented = ['blur', 'resolution', 'ghosting', 'motion', 'noise', 'spike']
                for a_idx, a_type in enumerate(augmented):
                    for idx, i in enumerate(range(4, 0, -1)): # Loop backwards through [1, 2, 3, 4] since the high numbers are of better quality
                        # Build new filename
                        a_filename = filename.split('.')[0] + '_' + a_type + str(i)
                        # Make directories
                        # Save augmented image (--> only) ? and original label ?
                        os.makedirs(os.path.join(target_path, a_filename, 'img'))
                        sitk.WriteImage(xs[a_idx+idx],
                            os.path.join(target_path, a_filename, 'img', "img.nii.gz"))

                        # Add original labels, that are not augmented --> Better discard this step ?
                        """
                        if not no_use_lfc:
                            os.makedirs(os.path.join(target_path, a_filename, 'seg'))
                            sitk.WriteImage(sitk.GetImageFromArray(y),
                                os.path.join(target_path, filename, 'seg', "001.nii.gz"))"""