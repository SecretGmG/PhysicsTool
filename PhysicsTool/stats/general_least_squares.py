import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
from typing import Callable

#for simplicity, only the most general iterative_least_squares function was added. It can always be applied. 
#Note that for simple cases (e.g. polynomial fit), one iteration will be sufficient.

#numpy polynomial fitting, no error analysis. 
def polynomial_fit(degree: int, x_data: np.array, y_data: np.array): 
    """calculate polynomial coefficients given degree and data, compute polynomial values
    use numpy functions for both tasks

    Args:
        degree (int): degree of polynomial fit
        x_data (np.array): independent data
        y_data (np.array): dependent data to fit polynomial in

    Returns:
        coeff (np.array): coefficients of polynomial fit
        poly_y_vals (np.array): polynomial values
    """
    coeff = np.polyfit(x_data, y_data, degree)
    poly_y_vals = np.polyval(coeff, x_data)

    return coeff, poly_y_vals

def iterative_least_squares(x0: np.array, data: np.array, observations: np.array, delta_obs: np.array, 
                            design_matrix_func, model_func, weight_matrix, epsilon: float, plot_func=None):
    """Generalized least squares regression for non-linear problems, iterative.
       Also calculates m0 at every iteration step and the cofactor matrix Q.

    Args:
        x0 (np.array): Initial parameters (as close as possible to the solution).
        data (np.array): Array containing the independent variable(s).
        observations (np.array): Array containing the dependent variable(s) (observations).
        delta_obs (np.array): Array containing the errors of the observations.
        design_matrix_func (function): Function to compute the design matrix based on current parameters.
        model_func (function): Function representing the model for the observations.
        epsilon (float): Maximum relative change between parameters for stopping criterion.
        plot_func (function, optional): Function to plot data during iterations. Defaults to None.

    Returns:
        x (np.array): Final estimation of the parameters.
        m0_array (np.array): Array containing m0 at every iteration.
        Q (np.array): Cofactor matrix.
    """
    # Calculate the initial weight matrix
    P = weight_matrix(delta_obs)
    
    iter = 0
    x = x0
    m0_array = []
    
    while True:
        # Calculate the design matrix with the current parameter estimate
        A = design_matrix_func(x, data)
        
        # Normal equation system matrix
        N = np.matmul(np.matmul(A.T, P), A)
        
        # Cofactor matrix
        Q = np.linalg.inv(N)
        
        # Residual vector
        delta_l = observations - model_func(data, x)
        
        # Parameter update
        delta_x = np.linalg.inv(A.T @ P @ A) @ (A.T @ P @ delta_l)
        x = x + delta_x
        
        # Stopping condition
        condition = np.max(np.abs(delta_x)) / np.linalg.norm(x + delta_x)
        
        # Increment iteration counter
        iter += 1
        
        # Calculate mean error of the weight unit a posteriori
        m0_squared = (delta_l.T @ P @ delta_l) / (len(observations) - len(x))
        m0 = np.sqrt(m0_squared)
        m0_array.append(m0)
        
        # Call the plot function if provided
        if plot_func:
            plot_func(data, observations, x, Q, m0_array[iter-1], iter-1)
        
        # Check stopping criteria
        if condition < epsilon or iter > 20:  # safety stop after 20 iterations
            return x, m0_array, Q
        
#some commonly used design matrix functions
def polynomial_design_matrix(x_vals : np.array, degree: int) -> np.array:
    """function that creates the design matrix for a polynomial of given degree and size

    Args:
        x_vals (array): array of x values from which we know the size of the design matrix
        degree (int): degree of the polynomial to fit into the data

    Returns:
        A (array): design matrix
    """
    m = len(x_vals)
    A = np.zeros((m, degree + 1))
    
    for i in range(m):
        for j in range(degree + 1):
            A[i, j] = x_vals[i] ** j   
    return A

#model test 
def model_test(data: np.array, final_x: np.array, alpha: float, m0: float, sigma0: float, residuals: np.array):
    """
    Performs a model test using the chi-square distribution to determine if the model fits the data well,
    incorporating considerations for m0 and sigma0.

    Args:
        data (np.array): Array of observed data.
        final_x (np.array): Array of estimated parameters from the model.
        alpha (float): Significance level for the test (e.g., 0.05 for a 5% level).
        m0 (float): Mean error of the weight unit a posteriori (standard deviation of residuals).
        sigma0 (float): Standard deviation of the measurement errors.
        residuals (np.array): Array of residuals from the model fit.

    Returns:
        test_statistic (float): Calculated chi-square test statistic.
        p_value (float): p-value for the test.
        critical_value (float): Critical value from chi-square distribution for the given alpha and degrees of freedom.
        pass_test (bool): True if the test statistic is less than the critical value, False otherwise.
    """
    # Calculate degrees of freedom
    f = len(data) - len(final_x)
    
    # Calculate test statistic, chi^2/(n-u)- distributed variable with expectation value 1
    test_statistic = np.sum((residuals / m0)**2) / sigma0**2
    
    # Calculate critical values from chi-square distribution
    critical_value_high = stats.chi2.ppf(1 - alpha, f)
    critical_value_low = stats.chi2.ppf(alpha, f)

    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(test_statistic, f)
    
    # Determine pass or fail
    pass_test = test_statistic < critical_value_high

    # Check m0 and sigma0 considerations, determine outcome of the test
    m0_over_sigma0_sq = (m0 ** 2) / (sigma0 ** 2)
    if np.isclose(m0_over_sigma0_sq, 1, atol=1e-2):
        status = "m0 and sigma0 are nearly equivalent. Model fit is acceptable."
    elif 1 < m0_over_sigma0_sq < critical_value_high:
        status = "Cautiously use m0 for comparisons. Test statistic indicates some discrepancy but is within a reasonable range."
    elif critical_value_low < m0_over_sigma0_sq < 1:
        status = "Use sigma0 for comparisons. Test statistic is below the critical value."
    else:
        status = "Investigate errors in the model. m0/sigma0 squared is outside the expected range."

    # Print results
    print(f"Degrees of Freedom (f): {f}")
    print(f"Test Statistic: {test_statistic:.4f}")
    print(f"Critical Value (High): {critical_value_high:.4f}")
    print(f"Critical Value (Low): {critical_value_low:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Pass Test: {'Yes' if pass_test else 'No'}")
    print(f"Status: {status}")

    return test_statistic, p_value, critical_value_high, pass_test, status