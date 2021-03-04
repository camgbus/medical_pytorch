# ------------------------------------------------------------------------------
# Dataset provided by JIP Tool.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import shutil
import torch
import traceback
import numpy as np
import SimpleITK as sitk
# Install lungmask from https://github.com/amrane99/lungmask using pip install git+https://github.com/amrane99/lungmask
from lungmask import mask
from mp.data.pytorch.transformation import centre_crop_pad_2d
from mp.data.datasets.dataset_cnn import CNNDataset, CNNInstance

class JIPDataset(CNNDataset):
    r"""Class for the dataset provided by the JIP tool/workflow.
    """
    def __init__(self, subset=None, img_size=(1, 299, 299), gpu=True, cuda=0, msg_bot=False):
        assert subset is None, "No subsets for this dataset."
        self.img_size = img_size
        self.gpu = gpu
        self.cuda = cuda
        self.msg_bot = msg_bot
        self.source_path = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"])
        self.dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"])

    def preprocess(self):
        # Delete data in directory and preprocess data.
        try:
            _delete_images(self.dataset_path)
            _extract_images(self.source_path, self.dataset_path, self.img_size, self.gpu, self.cuda)
            return True, None
        except: # catch *all* exceptions
            e = traceback.format_exc()
            return False, e

    def buildDataset(self):
        # Extract all images, if not already done
        if not os.path.isdir(self.dataset_path) or not os.listdir(self.dataset_path):
            _extract_images(self.source_path, self.dataset_path, self.img_size)


def _delete_images(path):
    # Walk through path and delete all .nii files
    print('Walk trough directory \'{}\' and delete nifti files..'.format(path))
    for dname, dirs, files in os.walk(path):
        for num, fname in enumerate(files):
            msg = str(num + 1) + ' of ' + str(len(files)) + ' file(s).'
            print (msg, end = '\r')
            # Check if file is a nifti file and delete it
            if '.nii' in fname:
                fpath = os.path.dirname(dname)
                shutil.rmtree(fpath)

def _extract_images(source_path, target_path, img_size=(1, 299, 299), gpu=False, cuda=0):
    r"""Extracts MRI images and saves the modified images."""
    # Foldernames are patient_id
    filenames = [x for x in os.listdir(source_path)]

    for num, filename in enumerate(filenames):
        msg = "Loading SimpleITK images/labels and center cropping them: "
        msg += str(num + 1) + " of " + str(len(filenames)) + "."
        print (msg, end = "\r")
        # Check if whole lung is captured
        discard, _, _ = _whole_lung_captured(os.path.join(source_path, filename, 'img', 'img.nii.gz'), gpu, cuda)
        
        if not discard:
            # Extract all images (3D)
            x = sitk.ReadImage(os.path.join(source_path, filename, 'img', 'img.nii.gz'))
            y = sitk.ReadImage(os.path.join(source_path, filename, 'seg', '001.nii.gz'))
            x = torch.from_numpy(sitk.GetArrayFromImage(x))
            y = torch.from_numpy(sitk.GetArrayFromImage(y).astype(np.int16))
            try:
                x = centre_crop_pad_2d(x, img_size)
                y = centre_crop_pad_2d(y, img_size)
            except:
                print('Image could not be resized and will therefore be skipped: {}.'
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

def _extract_lung_segmentation(input_path, gpu, cuda):
	# Load ct scan and create segmentation
	input_image = sitk.ReadImage(input_path)

    # load alternative models
	# model = mask.get_model('unet','LTRCLobes')
	# segmentation = mask.apply(input_image, model)

	segmentation = mask.apply(image=input_image, gpu=gpu, cuda=cuda.split(':')[-1])  # default model is U-net(R231)
	return segmentation

def _whole_lung_captured(input_path, gpu=True, cuda=0):
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