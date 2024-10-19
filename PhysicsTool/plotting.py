import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Optional

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def start_plt(title: str, xlabel: str, ylabel: str, grid: bool = True, ax=None, 
              fontsize: int = 13, labelsize: int = 12, ticksize: int = 11) -> None:
    """
    Set up the title, xlabel, ylabel, and grid for the matplotlib plot.
    
    Parameters:
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        grid (bool, optional): Whether to display grid lines on the plot. Defaults to True.
        ax (matplotlib.axes.Axes, optional): The axis to apply the settings to. Defaults to the current axis.
        fontsize (int, optional): Font size for the title. Defaults to 14.
        labelsize (int, optional): Font size for the axis labels. Defaults to 12.
        ticksize (int, optional): Font size for the tick labels. Defaults to 11.
    
    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)

    # Set the x and y axis tick label font sizes
    ax.tick_params(axis='x', labelsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)

    ax.grid(grid)

def end_plt(show: bool = True, legend_loc: str = 'best', legend_fontsize: int = 12, ax=None) -> None:
    """
    Finalize the matplotlib plot settings, display the legend, and optionally display the plot.
    
    Parameters:
        show (bool, optional): Whether to display the plot. Defaults to True.
        legend_loc (str, optional): The location of the legend on the plot. Defaults to 'best'.
        legend_fontsize (int, optional): Font size for the legend text. Defaults to 12.
        ax (matplotlib.axes.Axes, optional): The axis to apply the settings to. Defaults to the current axis.
    
    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()

    ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    if show:
        plt.show()

def err_band_plot(x: ArrayLike, y: ArrayLike, y_err: ArrayLike, label: Optional[str] = None, color: Optional[str] = None, ax: Optional[plt.Axes] = None) -> None:
    """
    Plots the function defined by x, y, and additionally the shaded error band according to y_err.

    Parameters:
        x (ArrayLike): The samples of the data.
        y (ArrayLike): The values corresponding to the samples.
        y_err (ArrayLike): The errors corresponding to the samples.
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
    plot, *_ = ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y_min, y_max, alpha=0.2, color=plot.get_color())

    # Optionally add a legend
    if label:
        ax.legend()
