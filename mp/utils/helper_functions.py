# ------------------------------------------------------------------------------
# Miscellaneous helper functions.
# ------------------------------------------------------------------------------

import numpy as np
from numbers import Number
import datetime
import ntpath

def f_optional_args(f, args, x):
    r"""If there are arguments, these are passed to the function."""
    if args:
        return f(x, **args)
    else:
        return f(x)

def get_time_string(cover=False):
    r"""
    Returns the current time in the format YYYY-MM-DD_HH-MM, or
    [YYYY-MM-DD_HH-MM] if 'cover' is set to 'True'.
    """
    date = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]
    if cover:
        return '['+date+']'
    else:
        return date

def divide_path_fname(path):
    r"""Divide path and name from a full path."""
    path_to_file, file_name = ntpath.split(path)
    if not file_name:
        # Cease where the path ends with a slash
        file_name = ntpath.basename(path_to_file)
        path_to_file = path_to_file.split(file_name)[0]
    return path_to_file, file_name

def average_dictionaries(*dicts):
    r"""For several dictionaries, averages the values for which they share
    the same keys into a new dictionary, which is returned."""
    mean_dict = dict()
    std_dict = dict()
    # Find common keys
    common_keys = set(dicts[0].keys())
    if len(dicts) > 1:
        for d in dicts:
            common_keys = common_keys.intersection(d.keys())
    # Average values
    for common_key in common_keys:
        values = [d[common_key] for d in dicts]
        if all(isinstance(val, dict) for val in values):  # Rec. apply function for dictionaries
            mean_dict[common_key], std_dict[common_key] = average_dictionaries(*values)
        elif all(isinstance(val, Number) for val in values):  # Average numeric values
            mean_dict[common_key] = np.mean(values)
            std_dict[common_key] = np.std(values)
    return mean_dict, std_dict