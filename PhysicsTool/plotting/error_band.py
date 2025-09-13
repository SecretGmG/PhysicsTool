import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Optional


def err_band_plot(
    x: ndarray,
    y: ndarray,
    y_err: ndarray,
    label: Optional[str] = None,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plots the function defined by x, y, and additionally the shaded error band according to y_err.

    Parameters:
        x (ndarray): The samples of the data"
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
