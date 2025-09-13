import numpy as np


def fft_to_coeffs(fft: np.ndarray, m=None) -> np.ndarray:
    """
    Converts the FFT coefficients to the coefficients of the polynomial.
    The coefficients are returned in the order a_0, a_1, ..., a_m, b_1, b_2, ... b_m
    """

    norm_fft = fft / len(fft)
    if m is None:
        m = len(norm_fft) // 2
    coeffs = np.zeros(2 * m + 1)
    coeffs[0] = norm_fft[0].real
    coeffs[1 : m + 1] = 2 * norm_fft[1 : m + 1].real
    coeffs[m + 1 :] = -2 * norm_fft[1 : m + 1].imag
    return coeffs


def coeffs_to_amplitude(coeffs: np.ndarray) -> np.ndarray:
    """
    Converts the coefficients of the polynomial to the amplitudes of the Fourier series.
    The coefficients are assumed to be in the order a_0, a_1, ..., a_m, b_1, b_2, ... b_m
    """
    m = len(coeffs) // 2
    assert len(coeffs) == 2 * m + 1, "coeffs must be of length 2*m+1"
    return np.concatenate(
        (coeffs[0:1], np.sqrt(coeffs[1 : m + 1] ** 2 + coeffs[m + 1 :] ** 2))
    )


def amplitude_spectrum(
    y: np.ndarray, m: int | None = None, d: float = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the amplitude spectrum of the data using numpy's fft.
    The amplitude is normalized by the length of the data, therefore corresponding to the factors of the discrete fourier transform.
    Returns:
        frequencies: Frequencies of the amplitude spectrum.
        amplitudes: Amplitudes of the amplitude spectrum.
    """
    if m is None:
        m = len(y) // 2
    fft = np.fft.fft(y)
    coeffs = fft_to_coeffs(fft, m)
    amplitudes = coeffs_to_amplitude(coeffs)
    frequencies = np.arange(m + 1) / (d * len(y))
    return frequencies, amplitudes


def discrete_fourier_transform(y: np.ndarray, m) -> np.ndarray:
    """
    Computes the discrete fourier transform of the data using a design matrix.
    The coefficients are returned in the order a_0, a_1, ..., a_m, b_1, b_2, ... b_m
    """
    x = np.arange(len(y))
    base_frequency = 2 * np.pi / len(y)
    # compute the design matrix
    A = np.column_stack(
        [np.cos(i * x * base_frequency) for i in range(0, m + 1)]
        + [np.sin(i * x * base_frequency) for i in range(1, m + 1)]
    )

    # compute the coefficients
    # coeffs = np.linalg.inv(A.T @ A) @ A.T @ y
    coeffs = np.linalg.lstsq(A, y)[0]

    return coeffs
