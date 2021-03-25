#-----------------------------------------------------------------
# A collection of tools in order to gather statistics on datasets
# that can then be used in order to spot faulty segmentation
#------------------------------------------------------------------

import numpy as np
import os
import mp.data.datasets.dataset_utils as du 
import torchio

def get_label_distribution_mean_and_std(dataset,renew_statistics=False,verbose=True):
    r"""Computes and saves or loads the computed mean and standard deviance of the fraction of the labels of the dataset appearing in the segmentations
    Args:
        dataset (mp.data.datasets.dataset_segmentation SegmentationDataset): the dataset we want to examine 
        renew_statistics (boolean): if mean and std need to be calculated again, e.g. when the dataset has changed 
        verbose (boolean): if the algorithm should print the results 
    Returns 
        mean (numpy.array)
        std (float)
    Examples:
        if the segmentations have 3 labels who are evenly distributed in each image, the result would be numpy.array([1/3,1/3,1/3]),0
    """
    global_name = dataset.name
    statistics_path = os.path.join('storage','statistics',global_name)

    label_names = dataset.label_names
    nr_labels = dataset.nr_labels
    size = dataset.size

    if not os.path.exists(statistics_path) or renew_statistics:
        image_densities = np.zeros((size,nr_labels))
        for i,instance in enumerate(dataset.instances):
            instance_num = instance.y.numpy()
            density,bin_edges = np.histogram(instance_num,bins=nr_labels,density=True)
            image_densities[i] = density*np.diff(bin_edges)
        mean = np.mean(image_densities,axis=0)
        std = np.std(image_densities,axis=0)
        if not os.path.exists(statistics_path):
            os.makedirs(statistics_path)
        np.save(os.path.join(statistics_path,'label_distribution_mean'), mean)
        np.save(os.path.join(statistics_path,'label_distribution_std'), std)
        if verbose:
            print('statistics computed and saved')
    else:
        mean = np.load(os.path.join(statistics_path,'label_distribution_mean.npy'),allow_pickle=True)
        std = np.load(os.path.join(statistics_path,'label_distribution_std.npy'),allow_pickle=True)
        if verbose:
            print('statistics loaded')
    
    if verbose:
        for i,label in enumerate(label_names):
            print('The mean appearance fraction of label {} is {:.4f} '.format(label,mean[i]))
        print('The standard deviance of the label appearance fractions is {:.4f} '.format(std[0]))
    
    return mean,std[0]