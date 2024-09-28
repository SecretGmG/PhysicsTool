import numpy as np
from .err import Err

def linregress(x_values, y_data: Err):
    """
    Performs a weighted least squares (WLS) linear regression, where weights are inversely proportional 
    to the variance (squared errors) of the dependent variable (y_data.err). The function computes 
    the regression slope, intercept, their standard errors, and returns the weighted means (centers) 
    of the independent and dependent variables.

    Parameters
    ----------
    x_values : np.ndarray
        1D array representing the independent variable (x-values).

    y_data : Err
        Instance of the `Err` class, containing:
        - mean (np.ndarray): The dependent variable (y-values).
        - err (np.ndarray): The uncertainties (errors) associated with the dependent variable.

    Returns
    -------
    tuple
        - slope (Err): An `Err` object representing the slope of the regression line and its standard error.
        - intercept (Err): An `Err` object representing the intercept of the regression line and its standard error.
        - x_center (float): The weighted mean of the independent variable (x-values).
        - y_center (float): The weighted mean of the dependent variable (y-values), using the weights from y_data.
    
    Notes
    -----
    The regression is performed using the formula:
    (X^T * W * X)^-1 * (X^T * W * y), where W is the diagonal weight matrix 
    derived from the inverse square of the errors (y_data.err).
    The standard errors of the slope and intercept are extracted from the 
    covariance matrix derived from the weighted residuals.
    """
    
    # Add a constant (intercept term) to the independent variable
    X = np.vstack((np.ones(x_values.shape[0]), x_values)).T  # Add column of ones for the intercept

    # Create diagonal weight matrix from the inverse square of errors
    weights = y_data.err ** -2
    W = np.diag(weights)

    # Perform Weighted Least Squares regression
    XtWX = np.dot(np.dot(X.T, W), X)  # X^T * W * X
    XtWy = np.dot(np.dot(X.T, W), y_data.mean)  # X^T * W * y
    regression_coefficients = np.linalg.inv(XtWX).dot(XtWy)  # (X^T * W * X)^-1 * (X^T * W * y)

    # Calculate residuals
    predicted_y_values = X.dot(regression_coefficients)
    residuals = y_data.mean - predicted_y_values

    # Calculate the variance of residuals
    residual_variance = np.sqrt((residuals.T @ (W / sum(W)) @ residuals) / (len(x_values) - 2))

    # Covariance matrix for regression coefficients
    cov_matrix = residual_variance * np.linalg.inv(XtWX)

    # Standard errors of the coefficients
    std_errors = np.sqrt(np.diag(cov_matrix))

    # Return the slope, intercept, and the weighted centers
    return (
        Err(regression_coefficients[1], std_errors[1]),  # slope and its error
        Err(regression_coefficients[0], std_errors[0]),  # intercept and its error
        np.average(x_values, weights=weights),  # weighted x_center
        y_data.weighted_average()  # weighted y_center (provided by Err class)
    )
