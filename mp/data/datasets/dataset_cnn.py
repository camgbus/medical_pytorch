# ------------------------------------------------------------------------------
# All datasets descend from this CNNDataset class storing cnn
# instances.
# ------------------------------------------------------------------------------

import os
import sys
from mp.data.datasets.dataset import Dataset, Instance
import mp.data.datasets.dataset_utils as du
import torchio
import torch

class CNNInstance(Instance):
    def __init__(self, x_path, y_label, name=None, class_ix=0, group_id=None):
        r"""A cnn instance, using the TorchIO library.

        Args:
        x_path (str): path to image
        y_label (torch.tensor): the intensity of noise in image at x_path (1, ..., 5)
        name (str): name of instance for case-wise evaluation
        class_ix (int): optional "class" index. During splitting of the dataset, 
            the resulting subsets are stratesfied according to this value (i.e. 
            there are about as many examples from each class in each fold
            of each class on each fold).
        group_id (comparable): instances with same group_id (e.g. patient id)
            are always in the same fold

        Note that torchio images have the shape (channels, w, h, d)
        """
        assert isinstance(x_path, str)
        assert torch.is_tensor(y_label) or y_label is None
        x = torchio.Image(x_path, type=torchio.INTENSITY)
        y = y_label
        self.shape = x.shape
        super().__init__(x=x, y=y, name=name, class_ix=class_ix, 
            group_id=group_id)

    def get_subject(self):
        return torchio.Subject(
            x=self.x,
            y=self.y
        )

class CNNDataset(Dataset):
    r"""A Dataset for cnn tasks, that specific datasets descend from.

        Args:
        instances (list[CNNInstance]): a list of instances
        name (str): the dataset name
        mean_shape (tuple[int]): the mean input shape of the data, or None
        label_names (list[str]): list with label names, or None
        nr_channels (int): number input channels
        modality (str): modality of the data, e.g. MR, CT
        hold_out_ixs (list[int]): list of instance index to reserve for a 
            separate hold-out dataset.
        check_correct_nr_labels (bool): Whether it should be checked if the 
            correct number of labels (the length of label_names) is consistent
            with the dataset. As it takes a long time to check, only set to True
            when initially testing a dataset.
    """
    def __init__(self, instances, name, mean_shape=None, 
    label_names=None, nr_channels=1, modality='unknown', hold_out_ixs=[],
    check_correct_nr_labels=False):
        # Set mean input shape and mask labels, if these are not provided
        print('\nDATASET: {} with {} instances'.format(name, len(instances)))
        if mean_shape is None:
            mean_shape, shape_std = du.get_mean_std_shape(instances)
            print('Mean shape: {}, shape std: {}'.format(mean_shape, shape_std))
        self.mean_shape = mean_shape
        self.label_names = label_names
        self.nr_labels = 0 if label_names is None else len(label_names)
        self.nr_channels = nr_channels
        self.modality = modality
        super().__init__(name=name, instances=instances, 
            mean_shape=mean_shape, output_shape=mean_shape, 
            hold_out_ixs=hold_out_ixs)