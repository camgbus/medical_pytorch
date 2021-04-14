import SimpleITK as sitk
# Install lungmask from https://github.com/amrane99/lungmask using pip install git+https://github.com/amrane99/lungmask
from lungmask import mask
import numpy as np

def _extract_lung_segmentation(input_path, gpu, cuda):
    r"""This function returns the lung segmentation of a CT scan."""
    # Load ct scan and create segmentation
    input_image = sitk.ReadImage(input_path)

    # load alternative models
    # model = mask.get_model('unet','LTRCLobes')
    # segmentation = mask.apply(input_image, model)

    segmentation = mask.apply(image=input_image, gpu=gpu, cuda=cuda.split(':')[-1])  # default model is U-net(R231)
    return segmentation

def whole_lung_captured(input_path, gpu=True, cuda=0):
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

    # Check now for each slice, if the lung is surrounded by non lung tissue,
    # ie. segmented tissue surrounded by zero values
    if not discard:
        for ct_slice in scan_np:
            # Check that top row from slice contains only 0
            if ct_slice[0].any():   # Contains at least one non zero value
                discard = True
                break
            # Check that bottom row from slice contains only 0
            if ct_slice[-1].any():  # Contains at least one non zero value
                discard = True
                break
            # Check that left column from slice contains only 0
            if ct_slice[:,0].any(): # Contains at least one non zero value
                discard = True
                break
            # Check that right column from slice contains only 0
            if ct_slice[:,-1].any():# Contains at least one non zero value
                discard = True
                break

    return discard, start_seg_idx, end_seg_idx