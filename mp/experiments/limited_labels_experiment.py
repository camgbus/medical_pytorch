# ------------------------------------------------------------------------------
# Experiment class that tracks experiments with different configurations.
# The idea is that if multiple experiments are performed, all intermediate
# stored files and model states are within a directory for that experiment. In
# addition, the experiment directory contains the config.json file with the
# original configuration, as well as the splitting of the dataset for each fold.
# When multiple repetitions, for instance within cross-validation, are
# performed, all files are within the experiment directory.
# ------------------------------------------------------------------------------

import os
import shutil
import time

import mp.utils.load_restore as lr
import mp.utils.pytorch.pytorch_load_restore as ptlr
from mp.data.data import Data
from mp.experiments.data_splitting import split_dataset, split_instances
from mp.paths import storage_path
from mp.utils.helper_functions import get_time_string
from mp.visualization.plot_results import plot_results
from .experiment import Experiment
from typing import Union, Dict
import numpy as np


class LimitedLabelsExperiment(Experiment):
    r"""
    A bundle of experiment runs with the same configuration
    for Domain Prediction with datasets with simulated limited labels
    (we actually have the labels, but don't want to use them for training).
    """

    def set_data_splits(self, data, limited_datasets=None):
        r"""
        Generates splits for the segmentor (train, test, val)
        and the domain predictor separately (train_dp, test_dp, val_dp)

        Args:
            limited_datasets (Union[None, Dict]): a dictionary of the form:
                                                  {dataset name: (nb labels training, nb labels validation)}
                                                  (is only used if data is of type Data)
        """
        if limited_datasets is None:
            limited_datasets = {}

        try:
            self.splits = lr.load_json(path=self.path, name='splits')
        except FileNotFoundError:
            print('Dividing dataset')
            # If the data consists of several datasets, then the splits are a
            # dictionary with one more label, that of the dataset name.
            if isinstance(data, Data):
                self.splits = dict()
                for ds_name, ds in data.datasets.items():
                    splits = split_dataset(ds, test_ratio=self.config.get('test_ratio', 0.0),
                                                             val_ratio=self.config['val_ratio'],
                                                             nr_repetitions=self.config['nr_runs'],
                                                             cross_validation=self.config['cross_validation'])
                    if ds_name in limited_datasets:
                        # Only a few data points have labels
                        # The splits that we have computed are for the domain predictor

                        # We just need to select the ixs from the train and val splits (test splits are the same)
                        nb_train, nb_val = limited_datasets[ds_name]
                        for split_run in splits:
                            # For some reason numpy casts the ints to int32
                            split_train = list(map(int, np.random.choice(split_run["train"],
                                                                         size=nb_train, replace=False)))
                            split_val = list(map(int, np.random.choice(split_run["val"],
                                                                       size=nb_val, replace=False)))

                            # First move the splits for the domain predictor
                            split_run["train_dp"] = split_run["train"]
                            split_run["val_dp"] = split_run["val"]
                            split_run["test_dp"] = split_run["test"].copy()  # Test splits are the same
                            # Then replace the train and val splits
                            split_run["train"] = split_train
                            split_run["val"] = split_val
                        self.splits[ds_name] = splits

                    else:
                        # All data points have labels
                        self.splits[ds_name] = splits
                        # The splits for the domain predictor are the same than the splits for the segmentor
                        for split_run in self.splits[ds_name]:
                            for split_name, split in list(split_run.items()):
                                split_run[split_name + "_dp"] = split.copy()

            else:
                self.splits = split_dataset(data, test_ratio=self.config.get('test_ratio', 0.0),
                                            val_ratio=self.config['val_ratio'], nr_repetitions=self.config['nr_runs'],
                                            cross_validation=self.config['cross_validation'])
            lr.save_json(self.splits, path=self.path, name='splits')
            print('\n')
