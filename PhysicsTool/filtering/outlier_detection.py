import numpy as np

def get_inliers(data : np.ndarray, f : float= 4.0, iterative : bool = True, robust = True):
    """
    Returns a boolean mask identifying inliers in `data` based on deviation from mean or median.

    Args:
        data: 1D data array.
        f: Threshold multiplier for deviation.
        iterative: If True, iteratively refines inliers.
        robust: If True, use robust stats (median & percentile).

    Returns:
        Boolean array marking inliers (True).
    """
    # start with the entire data as inliers
    inliers = np.ones_like(data, dtype=bool)

    while True:
        if robust:
            mean = np.median(data[inliers])
            deviations = np.abs(data[inliers] - mean)
            #use np.percentile for faster computation of the 68.3 percentile
            #it uses clever techniques to compute the percentile without sorting the entire array
            std = np.percentile(deviations, 68.3)
        else:
            std = data[inliers].std()
            mean = data[inliers].mean()

        new_inliers = np.abs(data-mean) < f*std

        # stop iterating if specified or if no inliers removed
        if (not iterative) or np.all(new_inliers == inliers):
            inliers = new_inliers
            break

        inliers = new_inliers

    return inliers
