import numpy as np
from typing import Literal
from .err import Err



def linear_linregress(x_values, y_data: Err, error_calculation: Literal['script', 'residuals', 'covariance']):
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
    
    assert not np.any(y_data.err == 0), 'Errors must be non-zero'
    
    if error_calculation == 'script':
        return practicum_script_linregress(x_values, y_data)
    
    
    X = np.column_stack([np.ones_like(x_values), x_values])
    features = weighted_least_squares_with_errors(X, y_data, error_calculation)
            
    
    
    return (
        features[1],  # slope and its error
        features[0],  # intercept and its error
        np.average(x_values, weights=y_data.err**-2),  # weighted x_center
        y_data.weighted_average()  # weighted y_center (provided by Err class)
    )
    
def weighted_least_squares_with_errors(X, y_data, error_calculation: Literal['residuals', 'covariance']):
    """
    Perform weighted least squares linear regression and return both
    the estimated coefficients and their standard errors.

    Parameters:
    X (np.ndarray): Design matrix (n_samples, n_features)
    y (np.ndarray): Target values (n_samples, 1)
    Sigma (np.ndarray): Covariance matrix of y (n_samples, n_samples)

    Returns:
    np.ndarray: Estimated coefficients (n_features, 1)
    np.ndarray: Standard errors of the estimated coefficients (n_features, 1)
    """

    y = y_data.mean.reshape(-1,1)
    Sigma = np.diag(y_data.err**2)
    
    # Compute the inverse of the covariance matrix Sigma
    Sigma_inv = np.linalg.inv(Sigma)

    # Calculate the weighted least squares solution
    Xt_Sigma_inv = np.dot(X.T, Sigma_inv)
    Xt_Sigma_inv_X_inv = np.linalg.inv(np.dot(Xt_Sigma_inv, X))
    beta_hat = Xt_Sigma_inv_X_inv.dot(Xt_Sigma_inv).dot(y)

    
    beta_cov_matrix = None
    match error_calculation:
        case 'covariance':
            beta_cov_matrix = Xt_Sigma_inv_X_inv
        case 'residuals':
            # Compute residuals: y_pred = X @ beta_hat
            y_pred = X @ beta_hat
            residuals = y - y_pred
            n = X.shape[0]  # Number of data points
            m = X.shape[1]  # Number of features
            rss_w = (residuals.T @ Sigma_inv @ residuals).item()  # Weighted RSS

            sigma_hat_squared = rss_w / (n - m)  # Variance term scaled by degrees of freedom
            beta_cov_matrix = Xt_Sigma_inv_X_inv*sigma_hat_squared
        

    # Compute the covariance matrix of beta_hat (incorporating Sigma and residual variance)

    # The standard errors are the square roots of the diagonal of the covariance matrix
    standard_errors = np.sqrt(np.diag(beta_cov_matrix))

    return Err(beta_hat[:,0], standard_errors)

def practicum_script_linregress(x, y: Err):
    n = len(x)
    denominator = n*sum(x**2) - sum(x)**2
    a = (n * sum(x*y.mean) - sum(x)*sum(y.mean)) / denominator
    b = (sum(y.mean)*sum(x**2) - sum(x)*sum(x*y.mean)) / denominator
    
    D = sum((a*x + b - y.mean)**2)
    s_y = np.sqrt(D / (n-2))
    s_a = np.sqrt(D/n) / np.std(x, ddof=1)
    
    return Err(a, s_a), Err(b, s_y), np.mean(x), np.mean(y.mean)
    
    