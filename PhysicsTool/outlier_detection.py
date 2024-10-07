import numpy as np

def outlier_finder(data_y: list, data_x: list, k: float = 3):
    """Removes outliers from the data (y axis) and corresponding x values as well.

    Args:
        data_y (list): Dependent variable data with possible outliers.
        data_x (list): Independent variable data corresponding to data_y.
        k (float): Multiple of standard deviation to define outlier sensitivity. Defaults to 3 (see Gauss rule). 

    Returns:
        non_outliers_y (list): List of dependent variable data with outliers removed.
        non_outliers_x (list): List of independent variable data corresponding to the final non-outliers in data_y.
    """
    median = np.median(data_y)
    std = np.std(data_y, ddof=1)
    lower_bound = median - k * std
    upper_bound = median + k * std

    # Find indices of outliers in data_y
    outlier_indices = [i for i, value in enumerate(data_y) if not (lower_bound <= value <= upper_bound)]
    
    # Remove outliers from data_y and corresponding elements from data_x
    non_outliers_y = [value for i, value in enumerate(data_y) if i not in outlier_indices]
    non_outliers_x = [value for i, value in enumerate(data_x) if i not in outlier_indices]

    return non_outliers_y, non_outliers_x


def iterative_outlier_finder(data_y: list, data_x: list, k: float, eps: float):
    """Iteratively removes outliers from the data, recalculates the median and standard deviation at each step.

    Args:
        data_y (list): Dependent variable data with possible outliers.
        data_x (list): Independent variable data corresponding to data_y.
        k (float): Multiple of standard deviation to define outlier sensitivity.
        eps (float): Convergence condition.

    Returns:
        final_non_outliers_y (list): List of dependent variable data with outliers removed after iterative processing.
        final_non_outliers_x (list): List of independent variable data corresponding to the final non-outliers in data_y.
    """
    all_outlier_indices = []
    
    median = np.median(data_y)
    std = np.std(data_y, ddof=1)
    
    lower_bound = median - k * std
    upper_bound = median + k * std
    
    non_outliers_y = data_y
    non_outliers_x = data_x
    new_median = np.median(non_outliers_y)

    while True:
        outlier_indices = [i for i, value in enumerate(non_outliers_y) if not (lower_bound <= value <= upper_bound)]
        
        if not outlier_indices:  # Stop if no outliers are found
            break
        
        all_outlier_indices.extend(outlier_indices)
        non_outliers_y = [value for i, value in enumerate(non_outliers_y) if i not in outlier_indices]
        non_outliers_x = [value for i, value in enumerate(non_outliers_x) if i not in outlier_indices]
        
        new_median = np.median(non_outliers_y)
        std = np.std(non_outliers_y, ddof=1)
        
        lower_bound = new_median - k * std
        upper_bound = new_median + k * std
        
        if abs(median - new_median) < eps:  # Convergence condition
            break
        
        median = new_median
    
    return non_outliers_y, non_outliers_x