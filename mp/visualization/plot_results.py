# ------------------------------------------------------------------------------
# Plots results.
# ------------------------------------------------------------------------------

import os
from typing import Iterable, Union

import matplotlib.pyplot as plt
import seaborn as sns

from mp.eval.result import Result
from mp.utils.seaborn.legend_utils import format_legend


def plot_results(result, measures=None, save_path=None, save_name=None,
                 title=None, ending='.png', ylog=False, figsize=(10, 5),
                 axvlines=None):
    """Plots a data frame as created by mp.eval.Results

    Args:
        result (Result): the Result object containing the data
        measures (list[str]): list of measure names
        save_path (str): path to save plot. If None, plot is shown.
        save_name (str): name with which plot is saved
        title (str): the title that will appear on the plot
        ending (str): can be '.png' or '.svg'
        ylog (bool): apply logarithm to y axis
        figsize (tuple[int]): figure size
        axvlines (Iterable[Union[Iterable, int]]): an iterable containing other iterables
                                                    (any depth permitted) or integers (for plotting vertical lines)

    """
    df = result.to_pandas()
    # Filter out measures that are not to be shown
    # The default is using all measures in the df
    if measures:
        df = df.loc[df['Metric'].isin(measures)]

    default_dashes = ["",
                      (4, 1.5),
                      (1, 1),
                      (3, 1, 1.5, 1),
                      (5, 1, 1, 1),
                      (5, 1, 2, 1, 2, 1),
                      (2, 2, 3, 1.5),
                      (1, 2.5, 3, 1.2)]

    # Start a new figure so that different plots do not overlap
    plt.figure()
    sns.set(rc={'figure.figsize': figsize})
    # Plot
    ax = sns.lineplot(x='Epoch',
                      y='Value',
                      hue='Metric',
                      style='Data',
                      alpha=0.7,
                      dashes=default_dashes,
                      data=df)
    ax = sns.scatterplot(x='Epoch',
                         y='Value',
                         hue='Metric',
                         style='Data',
                         alpha=1.,
                         data=df)

    # Plotting vertical lines
    if axvlines:
        def plot_vlines(iterable_or_int):
            if isinstance(iterable_or_int, int):
                plt.axvline(iterable_or_int, alpha=0.5, color="r", linestyle=":")
            else:
                for element in iterable_or_int:
                    plot_vlines(element)

        plot_vlines(axvlines)

    # Optional logarithmic scale
    if ylog:
        ax.set_yscale('log')
    # Style legend
    titles = ['Metric', 'Data']
    format_legend(ax, titles)
    # Set title
    if title:
        ax.set_title(title)
    # Save image
    if save_path:
        file_name = save_name if save_name is not None else result.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = file_name.split('.')[0] + ending
        plt.savefig(os.path.join(save_path, file_name), facecolor='w',
                    bbox_inches="tight", dpi=300)
