from typing import Self, Tuple
import numpy as np
from PhysicsTool.core.err import Err


class ErrFormat:
    '''
    A class for formatting error values and their means with specified precision and format settings.
    '''
    err_sigfigs: int
    val_sigfigs: int
    relative: bool
    exponent_factor: int
    min_positive_exponent: int
    max_negative_exponent: int

    def __init__(self, 
                 err_sigfigs: int = 2, 
                 val_sigfigs: int = 5, 
                 relative: bool = False, 
                 exponent_factor: int = 1,
                 min_positive_exponent: int = 3, 
                 max_negative_exponent: int = -3
                 ) -> None:
        '''
        Initialize the ErrFormat with settings for error precision, mean precision, relative error formatting, 
        and exponent factor.

        Args:
            err_sigfigs (int): Significant figures to display for the error, must be > 0.
            val_sigfigs (int): Significant figures to display for the value, must be > 0.
            relative (bool): Whether to display errors as relative percentages.
            exponent_factor (int): The factor used to determine exponent rounding, must be >= 0.
            min_positive_exponent (int): Minimum positive exponent for scientific notation.
            max_negative_exponent (int): Maximum negative exponent for scientific notation.
        '''
        if not all(isinstance(arg, int) and arg > 0 for arg in [err_sigfigs, val_sigfigs]):
            raise ValueError('err_sigfigs and val_sigfigs must be positive integers.')
        if not isinstance(relative, bool):
            raise ValueError('relative must be a boolean value.')
        if not isinstance(exponent_factor, int) or exponent_factor < 0:
            raise ValueError('exponent_factor must be a non-negative integer.')
        
        self.err_sigfigs = err_sigfigs
        self.val_sigfigs = val_sigfigs
        self.relative = relative
        self.exponent_factor = exponent_factor
        self.min_positive_exponent = min_positive_exponent
        self.max_negative_exponent = max_negative_exponent

    def combine(self, other: Self) -> Self:
        '''
        Combine this ErrFormat with another ErrFormat, taking the minimum of each setting.

        Args:
            other (Self): Another ErrFormat instance to combine with.

        Returns:
            ErrFormat: A new ErrFormat instance with combined settings.
        '''
        return ErrFormat(
            min(self.err_sigfigs, other.err_sigfigs),
            min(self.val_sigfigs, other.val_sigfigs),
            self.relative and other.relative,
            min(self.exponent_factor, other.exponent_factor),
            min(self.min_positive_exponent, other.min_positive_exponent),
            max(self.max_negative_exponent, other.max_negative_exponent)
        )

    def _calculate_exponent(self, val: float, error: float) -> int:
        '''Calculate the exponent based on value and error and the current exponent factor.'''
        if self.exponent_factor <= 0 or (val == 0 and error == 0):
            return 0
        relevant_value = val if val != 0 else error
        exponent = int(np.floor(np.log10(abs(relevant_value)) / self.exponent_factor) * self.exponent_factor)
        if exponent > 0 and exponent < self.min_positive_exponent:
            exponent = 0
        if exponent < 0 and exponent > self.max_negative_exponent:
            exponent = 0
        return exponent
            

    def _calculate_precision(self, error: float, mean_exponent: int) -> int:
        '''Calculate the precision based on error, error sigfigs, and mean exponent.'''
        if error != 0:
            error_exponent = int(np.floor(np.log10(abs(error))))
            return self.err_sigfigs - error_exponent - 1
        return self.val_sigfigs - mean_exponent - 1

    def _format(self, err: Err) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Format the means and errors in the given Err object according to the ErrFormat settings.

        Args:
            err (Err): The Err instance containing mean and error values.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of formatted mean values, errors, and exponents.
        '''
        mean_flat, err_flat = np.ravel(err.mean), np.ravel(err.err)
        
        formatted_means = np.empty_like(mean_flat, dtype=object)
        formatted_errs = np.empty_like(mean_flat, dtype=object)
        exponents = np.empty_like(mean_flat, dtype=int)

        for i, (val, error) in enumerate(zip(mean_flat, err_flat)):
            mean_exponent = self._calculate_exponent(val, error)

            mantissa_mean = val / (10.0 ** mean_exponent)
            mantissa_err = error / (10.0 ** mean_exponent)
            
            mantissa_precision = mean_exponent + self._calculate_precision(error, mean_exponent) 

            if mantissa_precision > 0:
                formatted_means[i] = f'{mantissa_mean:.{mantissa_precision}f}'
                formatted_errs[i] = f'{mantissa_err:.{mantissa_precision}f}'
            else:
                formatted_means[i] = f'{int(mantissa_mean):d}'
                formatted_errs[i] = f'{int(mantissa_err):d}'

            if self.relative:
                relative_error = 100 * mantissa_err / mantissa_mean
                relative_precision = self._calculate_precision(relative_error, 0)
                formatted_errs[i] = f'{relative_error:.{relative_precision}f}' if relative_precision > 0 else f'{int(relative_error):d}'

            exponents[i] = mean_exponent

        return (
            formatted_means.reshape(err.mean.shape),
            formatted_errs.reshape(err.err.shape),
            exponents.reshape(err.mean.shape)
        )

    def latex_array(self, err: Err, delimiter: str = '$') -> np.ndarray:
        '''
        Format the means and errors in the Err object as LaTeX strings.

        Args:
            err (Err): The Err instance containing mean and error values.
            delimiter (str): Delimiter for LaTeX math mode, default is '$'.

        Returns:
            np.ndarray: An array of formatted strings in LaTeX format.
        '''
        
        def format_to_latex(m, e, exp):
            percentage = r'\%' if self.relative else ''
            if exp != 0:
                return rf'{delimiter}({m} \pm {e}{percentage}) \times 10^{{{exp}}}{delimiter}'
            else:
                return rf'{delimiter}{m} \pm {e}{delimiter}{percentage}'
        
        
        formatted_mean, formatted_err, exponents = self._format(err)
        latex_strings = np.vectorize(format_to_latex)(formatted_mean, formatted_err, exponents)

        return latex_strings

    def string_array(self, err: Err) -> np.ndarray:
        '''
        Format the means and errors in the Err object as plain text strings.

        Args:
            err (Err): The Err instance containing mean and error values.

        Returns:
            np.ndarray: An array of formatted strings in plain text format.
        '''
        def format_to_string(m, e, exp):
            percentage = '%' if self.relative else ''
            if exp != 0:
                return f'({m} ± {e}{percentage}) × 10^{exp}'
            else:
                return f'{m} ± {e}{percentage}'
        
        formatted_mean, formatted_err, exponents = self._format(err)
        strings = np.vectorize(format_to_string)(formatted_mean, formatted_err, exponents)

        return strings

# Predefined ErrFormat instances for different formatting needs.
SCI_FORMAT = ErrFormat(1, 5, False, 1, 3, -3)
SCI_FORMAT_REL = ErrFormat(1, 5, True, 1, 3, -3)
ENG_FORMAT = ErrFormat(2, 5, False, 3, 3, -3)
ENG_FORMAT_REL = ErrFormat(2, 5, True, 3, 3, -3)
