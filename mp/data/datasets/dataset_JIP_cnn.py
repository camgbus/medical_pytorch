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
import mp.utils.load_restore as lr
from mp.data.pytorch.transformation import centre_crop_pad_3d
from mp.data.datasets.dataset_cnn import CNNDataset, CNNInstance
from mp.data.datasets.dataset_augmentation import augment_image_in_four_intensities as _augment_image
from mp.utils.lung_captured import whole_lung_captured as LungFullyCaptured
from mp.utils.generate_labels import generate_train_labels, generate_test_labels
from mp.data.datasets.dataset_utils import delete_images_and_labels

class JIPDataset(CNNDataset):
    r"""Class for the dataset provided by the JIP tool/workflow.
    """
    def __init__(self, subset=None, img_size=(1, 60, 299, 299), num_intensities=5, data_type='all', augmentation=True, data_augmented=False,
                 gpu=True, cuda=0, msg_bot=False, nr_images=20, build_dataset=False, dtype='train', noise='blur', ds_name='Decathlon', seed=42,
                 restore=False):
        r"""Constructor"""
        assert subset is None, "No subsets for this dataset."
        assert len(img_size) == 4, "Image size needs to be 4D --> (batch_size, depth, height, width)."
        self.img_size = img_size
        self.num_intensities = num_intensities
        self.augmentation = augmentation
        self.gpu = gpu
        self.cuda = cuda
        self.msg_bot = msg_bot
        self.data_type = data_type
        self.ds_name = ds_name
        self.nr_images = nr_images
        self.restore = restore
        self.data_augmented = data_augmented
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
                _extract_images(self.data_path, self.data_dataset_path, self.img_size, self.gpu, self.cuda, True, True, True)
                
            if self.data_type == 'train':
                if not self.restore:
                    delete_images_and_labels(self.train_dataset_path)
                    _extract_images(self.train_path, self.train_dataset_path, self.img_size, self.gpu, self.cuda)
                    _augment_extracted_images(self.train_dataset_path, self.img_size, False)    # Augmentation step without label consideration
                else:
                    _augment_extracted_images(self.train_dataset_path, self.img_size, False)    # Augmentation step without label consideration
                generate_train_labels(self.num_intensities, self.train_dataset_path, self.train_dataset_path, True)
                
            if self.data_type == 'test':
                delete_images_and_labels(self.test_dataset_path)
                _extract_images(self.test_path, self.test_dataset_path, self.img_size, self.gpu, self.cuda, True, True, False)
                generate_test_labels(self.num_intensities, self.test_dataset_path, self.test_dataset_path)

            if self.data_type == 'all':
                delete_images_and_labels(self.data_dataset_path)
                _extract_images(self.data_path, self.data_dataset_path, self.img_size, self.gpu, self.cuda, True, True, True)
                if not self.restore:
                    delete_images_and_labels(self.train_dataset_path)
                    _extract_images(self.train_path, self.train_dataset_path, self.img_size, self.gpu, self.cuda)
                    _augment_extracted_images(self.train_dataset_path, self.img_size, False)    # Augmentation step without label consideration
                else:
                    _augment_extracted_images(self.train_dataset_path, self.img_size, False)    # Augmentation step without label consideration
                generate_train_labels(self.num_intensities, self.train_dataset_path, self.train_dataset_path, True)
                delete_images_and_labels(self.test_dataset_path)
                _extract_images(self.test_path, self.test_dataset_path, self.img_size, self.gpu, self.cuda, True, True, False)
                generate_test_labels(self.num_intensities, self.test_dataset_path, self.test_dataset_path)

            return True, None
        except: # catch *all* exceptions
            e = traceback.format_exc()
            return False, e

    def buildDataset(self, dtype, noise, seed):
        r"""This function builds a dataset from the preprocessed (and augmented) data based on the dtype,
            either for training or inference. The dtype is the same as self.data_type, however it can not be
            'all' in this case, since it is important to be able to distinguish to which type a scan belongs
            (train -- inference). Noise specifies which data will be included in the dataset --> only used
            for training. ds_name specifies which dataset should be build, based on its name (in foldername).
            This can be 'Decathlon', 'GC' or 'FRA'. ds_name is only necessary for dtype == 'train'.
            NOTE: The function checks, if data is in the preprocessed folder, this does not mean, that it ensures
                  that the data is also augmented! If there is only preprocessed data (i.e. resampled and centre cropped),
                  then the preprocessing step should be performed again since this process includes the augmentation
                  (only for train data needed). In such a case, data_augmented in the config file should be set to False,
                  i.e. data is not augmentated and needs to be done."""
        # Extract all images, if not already done
        if dtype == 'train':
            if not os.path.isdir(self.train_dataset_path) or not os.listdir(self.train_dataset_path):
                print("Train data needs to be preprocessed..")
                self.data_type = dtype
                self.preprocess()
            if not self.data_augmented:
                _augment_extracted_images(self.train_dataset_path, self.img_size, False)    # Augmentation step without label consideration

        elif 'test' in dtype:
            if not os.path.isdir(self.test_dataset_path) or not os.listdir(self.test_dataset_path):
                print("Test data needs to be preprocessed..")
                self.data_type = 'test'
                self.preprocess()

        else:
            if not os.path.isdir(self.data_dataset_path) or not os.listdir(self.data_dataset_path):
                print("Inference data needs to be preprocessed..")
                self.data_type = dtype
                self.preprocess()

        # Assert if dtype is 'all'
        assert dtype != 'all', "The dataset type can not be all, it needs to be either 'train' or 'inference'!"

        # Build dataset based on dtype
        if dtype == 'inference':
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

        if dtype == 'train':            
            # Foldernames are patient_id
            study_names = [x for x in os.listdir(self.train_dataset_path) if 'DS_Store' not in x and '._' not in x]

            # Load labels and build one hot vector
            labels = lr.load_json(self.train_dataset_path, 'labels.json')
            one_hot = torch.nn.functional.one_hot(torch.arange(0, self.num_intensities), num_classes=self.num_intensities)

            # Load labels for selecting data equally distributed
            swapped_labels = lr.load_json(self.train_dataset_path, 'labels_swapped.json')

            # Build instances list
            instances = list()
            print()

            if self.ds_name == 'Decathlon' or self.ds_name == 'GC' or self.ds_name == 'FRA':
                study_names = _get_equally_distributed_names(study_names, swapped_labels, self.ds_name, noise, self.nr_images, self.num_intensities, seed)
                # Build instances
                for num, name in enumerate(study_names):
                    msg = 'Creating dataset from images: '
                    msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                    print (msg, end = '\r')
                    if 'Decathlon' not in name:
                        a_name = '_'.join(name.split('_')[:-1])
                    else:
                        a_name = name
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, a_name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[name] * self.num_intensities) - 1],
                        name = a_name,
                        group_id = None
                        ))

            elif self.ds_name == 'mixed':
                # Decathlon + GC + FRA
                Decathlon_names = _get_equally_distributed_names(study_names, swapped_labels, 'Decathlon', noise, self.nr_images, self.num_intensities, seed)
                GC_names = _get_equally_distributed_names(study_names, swapped_labels, 'GC', noise, self.nr_images, self.num_intensities, seed)
                FRA_names = _get_equally_distributed_names(study_names, swapped_labels, 'FRA', noise, self.nr_images, self.num_intensities, seed)
                study_names = Decathlon_names + GC_names + FRA_names

                # Build instances for Decathlon, GC and FRA
                for num, name in enumerate(study_names):
                    msg = 'Creating dataset from images: '
                    msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                    print (msg, end = '\r')
                    if 'Decathlon' not in name:
                        a_name = '_'.join(name.split('_')[:-1])
                    else:
                        a_name = name
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.train_dataset_path, a_name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[name] * self.num_intensities) - 1],
                        name = a_name,
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

        if 'test' in dtype:
            # Foldernames are patient_id based on dtype
            if dtype == 'testID':   # Split testID in seperate cases --> FRA, GC and Decathlon to split it in table...
                #study_names = [x for x in os.listdir(self.test_dataset_path) if 'DS_Store' not in x and '._' not in x\
                #               and 'GC' in x and '.json' not in x]
                study_names = [x for x in os.listdir(self.test_dataset_path) if 'DS_Store' not in x and '._' not in x\
                               and ('FRA' in x or 'GC' in x or 'Decathlon' in x)  and '.json' not in x]   # Use same data classes as trained on
            if dtype == 'testOOD':
                study_names = [x for x in os.listdir(self.test_dataset_path) if 'DS_Store' not in x and '._' not in x\
                               and 'FRA' not in x and 'GC' not in x and 'Decathlon' not in x and '.json' not in x]   # Use all other data classes as trained on
            
            # Load labels and build one hot vector
            labels = lr.load_json(self.test_dataset_path, 'labels.json')
            one_hot = torch.nn.functional.one_hot(torch.arange(0, self.num_intensities), num_classes=self.num_intensities)

            # Build instances, dataset without labels!
            instances = list()
            print()

            # Build instances
            for num, name in enumerate(study_names):
                msg = 'Creating test dataset from images: '
                msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                print (msg, end = '\r')
                a_name = name+'_'+noise
                instances.append(CNNInstance(
                    x_path = os.path.join(self.test_dataset_path, name, 'img', 'img.nii.gz'),
                    y_label = one_hot[int(labels[a_name] * self.num_intensities) - 1],
                    name = name,
                    group_id = None
                    ))

        return instances


def _get_equally_distributed_names(study_names, labels, ds_name, noise, nr_images, num_intensities, seed):
    r"""Extracts a list of folder names representing ds_name Dataset, based on noise and nr_images.
        An equal distribution of images will be extracted, ie. nr_images from each intensity level resulting
        in a dataset of num_intensities x nr_images foldernames."""
    # Set random seed
    random.seed(seed)

    # Select intensities equally
    ds_names = list()
    for i in range(1, num_intensities+1):
        labels_key = str(i/num_intensities) + '_' + noise
        possible_values = labels[labels_key]
        # Select only files from the current dataset with the current intensity level and where the name matches its label
        if ds_name == 'Decathlon':
            intensity_names = [x for x in possible_values if ds_name in x and x in study_names]
        else:
            intensity_names = [x for x in possible_values if ds_name in x and '_'.join(x.split('_')[:-1]) in study_names]

        # Select random names
        if len(intensity_names) > nr_images:
            ds_names.extend(random.sample(intensity_names, nr_images))
        else:
            ds_names.extend(intensity_names)

    # Reset random seed
    random.seed()
    return ds_names


def _extract_images(source_path, target_path, img_size=(1, 60, 299, 299), gpu=False, cuda=0,
                    no_use_lfc=False, no_crop_image=False, discard_labels=False):
    r"""This function extracts all CT scans from source path and resamples and centre crops them.
        The images will be saved at target_path.
        NOTE: no_use_lfc indicates if the LFC metrics should be considered, i.e. an image where the whole lung is not captured will not
        be used. no_crop_image is used to not crop the images based on img_size, but the unnecessary slices will be removed and
        only the dimensions of a slice will be cropped, not the final number of slices. discard_labels indicates if the labels should
        be considered as well, this is only necessary for the inference dataset, since there are no labels present and this would
        cause an error otherwise."""

    # Foldernames are patient_id
    filenames = [x for x in os.listdir(source_path) if 'DS_Store' not in x and '._' not in x]

    # Define resample object (each image will be resampled to voxelsize (1, 1, 3)) and transformation for centre cropping
    resample = tio.Resample((1, 1, 3))

    for num, filename in enumerate(filenames):
        msg = "Loading SimpleITK images/labels and center cropping them: "
        msg += str(num + 1) + " of " + str(len(filenames)) + "."
        print (msg, end = "\r")

        # Set default value for discard, in case for inference or test, where the LungFullyCaptured is not used!
        discard = False
        start_slc = 0
        end_slc = -1
        if not no_use_lfc:
            # Check if whole lung is captured
            discard, start_slc, end_slc = LungFullyCaptured(os.path.join(source_path, filename, 'img', 'img.nii.gz'), gpu, cuda)
        elif no_crop_image:
            # Extract start and end slice of lung to only crop unnecessary parts
            _, start_slc, end_slc = LungFullyCaptured(os.path.join(source_path, filename, 'img', 'img.nii.gz'), gpu, cuda)
     
        if not discard:
            # Extract all images (3D)
            try:
                x = resample(sitk.ReadImage(os.path.join(source_path, filename, 'img', 'img.nii.gz'))[:,:,start_slc:end_slc])
                x = torch.from_numpy(sitk.GetArrayFromImage(x)).unsqueeze_(0)
                if no_crop_image:
                    # Update img_size, for instance the number of slices
                    img_size = list(img_size)
                    if img_size[1] < x.size()[1]:
                        img_size[1] = x.size()[1]
                    img_size = tuple(img_size)
                if not discard_labels:
                    y = resample(sitk.ReadImage(os.path.join(source_path, filename, 'seg', '001.nii.gz'))[:,:,start_slc:end_slc])
                    y = torch.from_numpy(sitk.GetArrayFromImage(y).astype(np.int16)).unsqueeze_(0)
                    y = centre_crop_pad_3d(y, img_size)[0]
                x = centre_crop_pad_3d(x, img_size)[0]
            except:
                e = traceback.format_exc()
                print('Image could not be resampled/cropped and will therefore be skipped: {}.\nThe following error occured: {}.'
                .format(filename, e))
                continue

            # Create directories
            # Save new images so they can be loaded directly
            os.makedirs(os.path.join(target_path, filename, 'img'))
            sitk.WriteImage(sitk.GetImageFromArray(x), 
                os.path.join(target_path, filename, 'img', "img.nii.gz"))
            if not discard_labels:
                os.makedirs(os.path.join(target_path, filename, 'seg'))
                sitk.WriteImage(sitk.GetImageFromArray(y), 
                    os.path.join(target_path, filename, 'seg', "001.nii.gz"))


def _augment_extracted_images(source_path, img_size=(1, 60, 299, 299), consider_labels=False):
    r"""This function extracts all images from source_path and performs augmentation on them.
        Augmentation will be performed on Decathlon, FRA and GC data if desired. Afterwards, the images
        are centre cropped based on img_size. consider_labels indicated if the labels should be augmented
        as well. The new images will be stored under the same path, but with corresponding (different)
        names --> <foldername>_<augmentation_type><augmentation_intensity>.
        NOTE: This function relies on the pickle file called augmented_on.pkl stored under
              source_path. The file contains all scan names that have already been augmented. If this
              file does not exist, all scans will be augmented. --> Do not delete the generated file.
              With such an approach, the algorithm can continue where it stopped the time before, without
              doing augmentation on every single file again."""
    # Load list of already augmented scans
    augmented_on = lr.pkl_load('augmented_on.pkl', source_path)

    # Foldernames are patient_id (Exclude all augmented images)
    filenames = [x for x in os.listdir(source_path) if 'DS_Store' not in x and '._' not in x\
                 and 'blur' not in x and 'resolution' not in x and 'ghosting' not in x\
                 and 'noise' not in x and 'motion' not in x and 'spike' not in x]

    # Extract files that will be augmented
    files_to_aug = [x for x in filenames if 'Decathlon' in x or 'FRA' in x or 'GC' in x]
    if augmented_on is not None:
        files_to_aug = [x for x in files_to_aug if x not in augmented_on]
    else:   # File does not exist --> no augmentation done
        augmented_on = list()

    if len(files_to_aug) < 1:   # Nothing left to perform augmentation on
        return

    for num, filename in enumerate(files_to_aug):
        msg = "Loading SimpleITK images/labels and performing augmentation: "
        msg += str(num + 1) + " of " + str(len(files_to_aug)) + "."
        print (msg, end = "\r")
        
        # Extract all images (3D) --> Labels will not be cinsidered!
        x = sitk.ReadImage(os.path.join(source_path, filename, 'img', 'img.nii.gz'))
        x = torch.from_numpy(sitk.GetArrayFromImage(x)).unsqueeze_(0)
        if consider_labels:
            y = sitk.ReadImage(os.path.join(source_path, filename, 'seg', '001.nii.gz'))
            y = torch.from_numpy(sitk.GetArrayFromImage(y).astype(np.int16)).unsqueeze_(0)

        try: # Do augmentation
            xs = list()
            xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='blur'))
            xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='resolution'))
            xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='ghosting'))
            xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='motion'))
            xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='noise'))
            xs.extend(_augment_image(sitk.GetImageFromArray(x), noise='spike'))
            if consider_labels:
                xs.extend(_augment_image(sitk.GetImageFromArray(y), noise='blur'))
                xs.extend(_augment_image(sitk.GetImageFromArray(y), noise='resolution'))
                xs.extend(_augment_image(sitk.GetImageFromArray(y), noise='ghosting'))
                xs.extend(_augment_image(sitk.GetImageFromArray(y), noise='motion'))
                xs.extend(_augment_image(sitk.GetImageFromArray(y), noise='noise'))
                xs.extend(_augment_image(sitk.GetImageFromArray(y), noise='spike'))
            for idx, x_s in enumerate(xs):
                x_s = torch.from_numpy(sitk.GetArrayFromImage(x_s)).unsqueeze_(0)
                xs[idx] = centre_crop_pad_3d(x_s, img_size)[0]
        except:
            e = traceback.format_exc()
            print('Image could not be resized/resampled/augmented and will therefore be skipped: {}.\nThe following error occured: {}.'
            .format(filename, e))
            continue

        # Create directories
        # Save new images so they can be loaded directly
        augmented = ['blur', 'resolution', 'ghosting', 'motion', 'noise', 'spike']
        for a_idx, a_type in enumerate(augmented):
            for idx, i in enumerate(range(4, 0, -1)): # Loop backwards through [1, 2, 3, 4] since the high numbers are of better quality
                # Build new filename
                a_filename = filename.split('.')[0] + '_' + a_type + str(i)
                # Make directories
                # Save augmented image
                os.makedirs(os.path.join(source_path, a_filename, 'img'))
                sitk.WriteImage(sitk.GetImageFromArray(xs[a_idx+idx]),
                    os.path.join(source_path, a_filename, 'img', "img.nii.gz"))

                if consider_labels:
                # Add original labels, that are augmented as well, if desired
                    os.makedirs(os.path.join(source_path, a_filename, 'seg'))
                    sitk.WriteImage(sitk.GetImageFromArray(xs[a_idx+idx+24]),   # Labels are at offset 6 * 4 --> 24
                        os.path.join(source_path, a_filename, 'seg', "001.nii.gz"))
        
        # Add augmented image to augmented_on list and save the file (replace old one)
        augmented_on.append(filename)
        lr.pkl_dump(augmented_on, 'augmented_on.pkl', source_path)