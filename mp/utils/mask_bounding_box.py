import numpy as np


def mask_bbox_3D(img):
    """
    Returns the bounding box for the non-zero values in a 3D array as a (3, 2) tuple
    Inspired by: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    Args:
        img (np.array): a 3D array
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return (rmin, rmax), (cmin, cmax), (zmin, zmax)
