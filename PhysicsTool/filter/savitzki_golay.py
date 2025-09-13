import numpy as np
from typing import Literal


def savitzki_golay_filter(
    y: np.ndarray,
    deg: int = 1,
    window: int | None = None,
    edge: Literal["pad", "polynomial", "none"] = "polynomial",
) -> np.ndarray:
    """
    Applies a Savitzki-Golay filter to the input data y.
    Args:
        y: Input data to be filtered.
        deg: Degree of the polynomial to fit, default is 1 (linear).
        window: Size of the window to use for the filter, must be odd. If None, it will be set to deg*2+1.
        edge: How to handle the edges of the data. Options are 'pad', 'polynomial', or 'none'.
            'pad' will pad the data with the edge values,
            'polynomial' will use the polynomial coefficients of the first and last window to compute the start and end of the filtered signal.
            'none' will not handle the edges and return only the valid part of the convolution. in this case the return size will be len(y) - window + 1.
    Returns:
        filtered: The filtered data.
    """

    if window is None:
        window = deg * 2 + 1

    assert window % 2 == 1, "Window size must be odd"

    radius = window // 2

    A = np.array([[j**i for i in range(deg + 1)] for j in range(-radius, radius + 1)])
    B = np.linalg.pinv(A.T @ A) @ A.T

    match edge:
        case "pad":
            y = np.pad(y, (radius, radius), mode="edge")
            filtered = np.convolve(B[0, :], y, mode="valid")
        case "polynomial":
            # use the polynomial coefficients to compute the start and end of the filtered signal
            unproblematic_part = np.convolve(B[0, :], y, mode="valid")
            start_params = B @ y[:window]
            end_params = B @ y[-window:]
            start = np.polyval(start_params[::-1], np.arange(-radius, 0))
            end = np.polyval(end_params[::-1], np.arange(1, radius + 1))
            filtered = np.concatenate((start, unproblematic_part, end))
        case "none":
            filtered = np.convolve(B[0, :], y, mode="valid")

    return filtered
