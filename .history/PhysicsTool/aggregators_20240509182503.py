import numpy as np
from typing import Tuple

def err_from_data(array: np.ndarray, axis=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and standard error of the mean from data.

    Parameters:
        array (np.ndarray): The input data array.
        axis (Optional[int]): The axis along which the mean and error are calculated.
            If None, calculates the mean and error over the entire array. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The mean and standard error of the mean.
    """
    return np.mean(array, axis=axis), np.std(array, axis=axis) / np.sqrt(np.size(array, axis=axis))


def avg_err(mean: np.ndarray, err: np.ndarray, axis=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the average and combined error from means and errors.

    Parameters:
        mean (np.ndarray): The array of mean values.
        err (np.ndarray): The array of errors corresponding to the mean values.
        axis (Optional[int]): The axis along which the average and error are calculated.
            If None, calculates the average and error over the entire array. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The weighted average and combined error.
    """
    return np.mean(mean, axis=axis), np.sqrt(np.sum(err**2, axis=axis)) / np.size(err, axis=axis)

def weighted_avg_err(mean: np.ndarray, err: np.ndarray, axis=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the weighted average and combined error from means and errors. As explained here:
    https://www.physics.umd.edu/courses/Phys261/F06/ErrorPropagation.pdf

    Parameters:
        mean (np.ndarray): The array of mean values.
        err (np.ndarray): The array of errors corresponding to the mean values.
        axis (Optional[int]): The axis along which the average and error are calculated.
            If None, calculates the average and error over the entire array. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The weighted average and combined error.
    """
    return np.sum(mean*err**-2, axis=axis)/np.sum(err**-2), np.sum(err**-2, axis=axis)**-0.5
