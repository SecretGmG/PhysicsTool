from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from numpy import ndarray


def set_theme():
    """Set a consistent theme for scientific matplotlib plots."""
    matplotlib.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "legend.fontsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "figure.dpi": 200,
        "figure.autolayout": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def set_up_plot(
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    ax=None,
) -> None:
    """
    Set up the title, xlabel, ylabel, and grid for the matplotlib plot.

    Parameters:
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        ax (matplotlib.axes.Axes, optional): The axis to apply the settings to. Defaults to the current axis.

    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()
    if title:
        ax.set_title(title)  # type: ignore
    ax.set_xlabel(xlabel)  # type: ignore
    ax.set_ylabel(ylabel)  # type: ignore

def err_band_plot(
    x: ndarray,
    y: ndarray,
    y_err: ndarray,
    label: Optional[str] = None,
    color: Optional[str] = None,
    ax=None,
) -> None:
    """
    Plots the function defined by x, y, and additionally the shaded error band according to y_err.

    Parameters:
        x (ndarray): The samples of the data.
        y (ndarray): The values corresponding to the samples.
        y_err (ndarray): The errors corresponding to the samples.
        label (Optional[str]): The label for the plot.
        color (Optional[str]): The color of the plot.
        ax (Optional[plt.Axes]): The axes to plot on. Defaults to the current axes.

    Returns:
        None
    """
    # Initialize ax to current axes if not provided
    if ax is None:
        ax = plt.gca()

    # Calculate the upper and lower bounds for the error band
    y_min = y - y_err
    y_max = y + y_err

    # Create the plot
    plot, *_ = ax.plot(x, y, color=color, label=label)  # type: ignore
    ax.fill_between(x, y_min, y_max, alpha=0.2, color=plot.get_color())  # type: ignore

    # Optionally add a legend
    if label:
        ax.legend()  # type: ignore
