import matplotlib.pyplot as plt

def start_plt(title: str, xlabel: str, ylabel: str, grid: bool = True) -> None:
    """
    Set up the title, xlabel, ylabel, and grid for the matplotlib plot.

    Parameters:
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        grid (bool, optional): Whether to display grid lines on the plot. Defaults to True.

    Returns:
        None
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)


def end_plt(show: bool = True, legend_loc: str = 'best') -> None:
    """
    Finalize the matplotlib plot settings, display the legend, and optionally display the plot.

    Parameters:
        show (bool, optional): Whether to display the plot. Defaults to True.
        legend_loc (str, optional): The location of the legend on the plot. Defaults to 'best'.

    Returns:
        None
    """
    plt.legend(loc=legend_loc)
    if show:
        plt.show()

def err_band_plot(x, y, y_err, label = None, color = None):
    """
    Plots the function defined by x, y, and additionally the a shaded error band according to y_err

    Parameters:
        x ArrayLike: The samples of the data.
        y ArrayLike: The values corresponding to the samples
        y_err ArrayLike: The errors corresponding to the samples
        

    Returns:
        None
    """
    y_min = y-y_err
    y_max = y+y_err
    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y_min, y_max, alpha=0.2, color=color)