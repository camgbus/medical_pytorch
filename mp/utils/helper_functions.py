# ------------------------------------------------------------------------------
# Miscellaneous helper functions.
# ------------------------------------------------------------------------------

def f_optional_args(f, args, x):
    r"""If there are arguments, these are passed to the function."""
    if args:
        return f(x, **args)
    else:
        return f(x)

import datetime
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

import ntpath

def divide_path_fname(path):
    r"""Divide path and name from a full path."""
    path_to_file, file_name = ntpath.split(path)
    if not file_name:
        # Cease where the path ends with a slash
        file_name = ntpath.basename(path_to_file)
        path_to_file = path_to_file.split(file_name)[0]
    return path_to_file, file_name


import numpy as np


def zip_longest_with_cycle(*iterators):
    r"""Combines zip_longest with cycles for the shorter iterators
    (assumes that you can iterate multiple times over each iterator to avoid saving the elements
    and spare memory)"""
    iters_copy = [iter(iterator) for iterator in iterators]
    finished = [False] * len(iters_copy)

    while True:
        next_elements = []
        for idx, iterator in enumerate(iters_copy):
            # For each iterator, get the next element and cycle iterator if needed
            ele = next(iterator, None)
            if ele is None:
                # The current iterator is finished, create another one and flag it as finished at least ince
                finished[idx] = True
                iters_copy[idx] = iter(iterators[idx])
                ele = next(iters_copy[idx])
            next_elements.append(ele)

        if np.alltrue(finished):
            break

        yield tuple(next_elements)


