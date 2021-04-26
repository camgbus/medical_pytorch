import os
import SimpleITK as sitk
import numpy as np
from itertools import combinations
# Install surface_distance from https://github.com/amrane99/surface-distance using pip install git+https://github.com/amrane99/surface-distance
from surface_distance import metrics

def calculateSegmentationsSimilarity(segmentations, surfaceThreshold):
    r"""This function takes a list of paths to segmentations (of the same object but
        but different observers) and calculates the surface dice coefficient of each
        possible segmentation pair given the surfaceThreshold (in mm). At the end, the mean
        of all calculated scores will be returned.
        E.g.: 3 segmentation (1, 2, 3) --> calculate score for (1, 2) - (1, 3) - (2, 3)
              and return the mean of the three calculated scores.
    """
    # Extract the number of segmentations
    nr_segmentations = len(segmentations)
    # Initialize dict for arrays and list for scores
    segmentations_dict = dict()
    scores = list()

    # Extract the spacing from the first element of the list
    reader = sitk.ImageFileReader()
    reader.SetFileName(segmentations[0])
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    spacing = [float(reader.GetMetaData('pixdim[1]')), float(reader.GetMetaData('pixdim[2]')), float(reader.GetMetaData('pixdim[3]'))]

    # Load images as SimpleITK, than arrays
    for idx, path in enumerate(segmentations):
        segmentations_dict[idx] = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(bool)

    # Permutate all possible pairs based on nr_segmentations
    comb = combinations(range(nr_segmentations), 2)

    # Loop through combinations and calculate the surface dice score
    for combination in comb:
        distances = metrics.compute_surface_distances(segmentations_dict[combination[0]], segmentations_dict[combination[1]], spacing)
        scores.append(metrics.compute_surface_dice_at_tolerance(distances, surfaceThreshold))

    # Return the mean score based on scores
    mean_score = sum(scores) / len(scores)
    return mean_score


if __name__ == '__main__':
    segmentations = ['path_to_seg_1.nii.gz',
                     'path_to_seg_2.nii.gz',
                     # ...
                     'path_to_seg_n.nii.gz']  # Change accordingly
    surfaceThreshold = 2  # in mm! --> Change accordingly
    similarity = calculateSegmentationsSimilarity(segmentations, surfaceThreshold)
    print(similarity)
