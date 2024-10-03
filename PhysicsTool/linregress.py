import numpy as np
from typing import Literal, Tuple
from .err import Err


def linear_linregress(x_values: np.ndarray, y_data: Err, error_calculation: Literal['script', 'residuals', 'covariance']) -> Tuple[Err, Err, float, float]:
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
        
    error_calculation : {'script', 'residuals', 'covariance'}
    
        Method to calculate the uncertainties (errors) in the regression coefficients (slope and intercept).
    
        - 'script': 
            Uses a pre-defined method for error calculation based on the practicum script logic. 
            This is a simpler, less flexible option that does not account for weighted errors.
            It's typically used for academic exercises or where errors are not a key concern.
            It performs basic linear regression using a closed-form formula and returns standard 
            errors for the slope and intercept without considering weighted least squares (WLS).
        
        - 'residuals': 
            Uses the residuals of the regression (i.e., the difference between observed and predicted values) 
            to estimate the variance of the coefficients. The method calculates the residual sum of squares (RSS) 
            and scales the covariance matrix of the regression coefficients by the variance derived from the residuals. 
            This is useful when you want to account for the fit's quality and adjust uncertainties based on residuals.
        
        - 'covariance': 
            Uses the covariance matrix of the dependent variable's errors (`y_data.err`) directly in the calculation 
            of the regression coefficients' uncertainties. The covariance matrix is treated as a known and fixed 
            quantity. This method is useful when you have well-characterized errors in the dependent variable and 
            want to propagate them directly through the regression.

    Returns
    -------
    tuple
        - slope (Err): An `Err` object representing the slope of the regression line and its standard error.
        - intercept (Err): An `Err` object representing the intercept of the regression line and its standard error.
        - x_center (float): The weighted mean of the independent variable (x-values).
        - y_center (float): The weighted mean of the dependent variable (y-values), using the weights from y_data.
    """
    
    assert not np.any(y_data.err == 0), 'Errors must be non-zero'
    
    if error_calculation == 'script':
        return practicum_script_linregress(x_values, y_data)

    X = np.column_stack([np.ones_like(x_values), x_values])

    # Get slope and intercept using the weighted least squares method
    slope, intercept = weighted_least_squares_with_errors(X, y_data, error_calculation)
    
    # Compute weighted means (centers)
    x_center = np.average(x_values, weights=1 / y_data.err**2)
    y_center = y_data.weighted_average()

    return slope, intercept, x_center, y_center


def weighted_least_squares_with_errors(X: np.ndarray, y_data: Err, error_calculation: Literal['residuals', 'covariance']) -> Tuple[Err, Err]:
    """
    Perform weighted least squares linear regression and return both
    the estimated coefficients and their standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_samples, n_features).
    y_data : Err
        Target values wrapped in an `Err` object (n_samples, 1).

    Returns
    -------
    Tuple[Err, Err]
        - slope: The slope of the regression line with its error.
        - intercept: The intercept of the regression line with its error.
    """

    y = y_data.mean.reshape(-1, 1)
    Sigma = np.diag(y_data.err**2)
    
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular and cannot be inverted.")

    Xt_Sigma_inv = X.T @ Sigma_inv
    Xt_Sigma_inv_X_inv = np.linalg.inv(Xt_Sigma_inv @ X)
    beta_hat = Xt_Sigma_inv_X_inv @ Xt_Sigma_inv @ y

    beta_cov_matrix = None
    match error_calculation:
        case 'covariance':
            beta_cov_matrix = Xt_Sigma_inv_X_inv
        case 'residuals':
            y_pred = X @ beta_hat
            residuals = y - y_pred
            n, m = X.shape
            rss_w = residuals.T @ Sigma_inv @ residuals
            sigma_hat_squared = rss_w.item() / (n - m)
            beta_cov_matrix = Xt_Sigma_inv_X_inv * sigma_hat_squared

    # Extract standard errors from the covariance matrix
    standard_errors = np.sqrt(np.diag(beta_cov_matrix))

    return Err(beta_hat[1, 0], standard_errors[1]), Err(beta_hat[0, 0], standard_errors[0])


def practicum_script_linregress(x: np.ndarray, y: Err) -> Tuple[Err, Err, float, float]:
    """
    Perform a basic linear regression using a simple formula without weight-based error handling.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : Err
        Dependent variable wrapped in an `Err` object.

    Returns
    -------
    Tuple[Err, Err, float, float]
        - Slope with error (Err).
        - Intercept with error (Err).
        - Mean of x.
        - Mean of y.
    """
    n = len(x)
    denominator = n * np.sum(x**2) - np.sum(x)**2
    slope = (n * np.sum(x * y.mean) - np.sum(x) * np.sum(y.mean)) / denominator
    intercept = (np.sum(y.mean) * np.sum(x**2) - np.sum(x) * np.sum(x * y.mean)) / denominator

    D = np.sum((slope * x + intercept - y.mean) ** 2)
    s_y = np.sqrt(D / (n - 2))
    s_slope = np.sqrt(D / n) / np.std(x, ddof=1)

    return Err(slope, s_slope), Err(intercept, s_y), np.mean(x), np.mean(y.mean)
