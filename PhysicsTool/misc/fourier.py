import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats

#used to calculate Fourier series with custom m values, based on the least squares method
def A_matrix(time: np.array, m: int) -> np.array:
    """design matrix for the fourier problem 

    Args:
        time (np.array): time values
        m (int): maximal development degree of fourier series

    Returns:
        A (np.array): design matrix
    """
    A = np.zeros((len(time), 2 * m + 1))  
    omega = 2 * np.pi / abs(time[0] - time[-1])
    for i in range(len(time)):
        row = np.zeros(2 * m + 1)
        row[0] = 1  # The constant term a0
        for m_idx in range(1, m + 1):
            row[2 * m_idx - 1] = np.cos(m_idx * omega * time[i])
            row[2 * m_idx] = np.sin(m_idx * omega * time[i])
        A[i] = row
    return A

def fourier_coefficients(acc: np.array, time: np.array, m: int)-> np.array:
    """calculate the least squares solution using the design matrix A

    Args:
        acc (np.array): linear accelerations in S direction values
        time (np.array): time array
        m (int): maximal development degree of fourier series

    Returns:
        x (np.array): vector containing the fourier coefficients
    """
    A = A_matrix(time, m)
    x = np.linalg.inv(A.T@A)@A.T@acc
    return x

def fourier_function(coefficients: np.array, time: np.array)-> np.array:
    """use the calculated fourier coefficients to calculate the fourier function

    Args:
        coefficients (np.array): fourier coefficients
        time (np.array): array of time data

    Returns:
        function (np.array): calculated fourier function 
    """
    omega = 2 * np.pi / abs(time[0] - time[-1])
    m = (len(coefficients) - 1) // 2
    a0 = coefficients[0]
    a_coeff = coefficients[1:2*m:2]
    b_coeff = coefficients[2:2*m+1:2]
    
    function = np.zeros(len(time))
    for j in range(len(time)):
        function[j] = a0 + np.sum([a_coeff[i] * np.cos((i + 1) * omega * time[j]) for i in range(m)]) + np.sum([b_coeff[i] * np.sin((i + 1) * omega * time[j]) for i in range(m)])
    
    return function

#amplitude and power spectra
def spectrum(data: np.array, time: np.array):
    """calculates the amplitude and power spectrums using the numpy fft function

    Args:
        data (np.array): amplitude data on the y axis/ measurement values
        time (np.array): time data
    """
    N = len(data)
    dt = time[0] - time[-1]
    freq = np.fft.fftfreq(N)*N/dt
    freq = np.abs(freq)[:N//2]

    data_ddt = np.fft.fft(data)
    amplitude_data = 2/N*np.abs(data_ddt)
    amplitude_data = amplitude_data[:N//2]
    
    return freq, amplitude_data
