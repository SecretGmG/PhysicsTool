import numpy as np
from .err import Err

def linregress(x_values, y_data: Err):
    """
    Performs a weighted least squares (WLS) linear regression using weights inversely proportional to
    the variance of the dependent variable (y_data.err)^2.
    
    Parameters:
    x_values : np.ndarray
        The independent variable data (e.g., x values).
    y_data : Err
        An object of class Err containing the dependent variable means (y_data.mean) and their associated
        uncertainties (y_data.err).
    
    Returns:
    tuple of Err
        Returns a tuple containing two Err objects:
        - (slope, slope_err): The slope of the regression line and its associated standard error.
        - (intercept, intercept_err): The intercept of the regression line and its associated standard error.
    """
    
    # Perform weighted least squares regression
    # Add a constant (intercept term) to the independent variable
    X = np.vstack((np.ones(x_values.shape[0]), x_values)).T  # Add column of ones for the intercept

    # Create diagonal weight matrix
    weights = y_data.err**-2
    W = np.diag(weights)

    # Perform Weighted Least Squares regression
    # Formula: (X^T * W * X)^-1 * (X^T * W * y)
    XtWX = np.dot(np.dot(X.T, W), X)  # X^T * W * X
    XtWy = np.dot(np.dot(X.T, W), y_data.mean)  # X^T * W * y
    beta_hat = np.linalg.inv(XtWX).dot(XtWy)  # (X^T * W * X)^-1 * (X^T * W * y)

    # Calculate residuals
    y_hat = X.dot(beta_hat)
    residuals = y_data.mean - y_hat

    # Calculate variance of residuals
    sigma_hat = np.sqrt((residuals.T @ (W / sum(W)) @ residuals) / (len(x_values)-2))

    # Calculate covariance matrix for beta_hat
    cov_matrix = sigma_hat * np.linalg.inv(XtWX)

    # Standard errors of the coefficients
    std_errors = np.sqrt(np.diag(cov_matrix))

    return Err(beta_hat[1], std_errors[1]), Err(beta_hat[0], std_errors[0])