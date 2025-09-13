from typing import Tuple
import numpy as np
from copy import deepcopy


class UErrFormatter:
    """
    A class for formatting error values and their means with specified precision and format settings.
    """

    err_sigfigs: int
    val_sigfigs: int
    relative: bool
    exponent_factor: int
    min_positive_exponent: int
    max_negative_exponent: int
    use_siunitx: bool

    def __init__(
        self,
        err_sigfigs: int = 1,
        val_sigfigs: int = 5,
        relative: bool = False,
        exponent_factor: int = 1,
        min_positive_exponent: int = 3,
        max_negative_exponent: int = -3,
        use_siunitx: bool = True,
    ) -> None:
        """
        Initialize the ErrFormat with settings for error precision, mean precision, relative error formatting,
        and exponent factor.

        Args:
            err_sigfigs (int): Significant figures to display for the error, must be > 0.
            val_sigfigs (int): Significant figures to display for the value, must be > 0.
            relative (bool): Whether to display errors as relative percentages.
            exponent_factor (int): The factor used to determine exponent rounding, must be >= 0.
            min_positive_exponent (int): Minimum positive exponent for scientific notation.
            max_negative_exponent (int): Maximum negative exponent for scientific notation.
        """
        if not all(
            isinstance(arg, int) and arg > 0 for arg in [err_sigfigs, val_sigfigs]
        ):
            raise ValueError("err_sigfigs and val_sigfigs must be positive integers.")
        if not isinstance(relative, bool):
            raise ValueError("relative must be a boolean value.")
        if not isinstance(exponent_factor, int) or exponent_factor < 0:
            raise ValueError("exponent_factor must be a non-negative integer.")

        self.err_sigfigs = err_sigfigs
        self.val_sigfigs = val_sigfigs
        self.relative = relative
        self.exponent_factor = exponent_factor
        self.min_positive_exponent = min_positive_exponent
        self.max_negative_exponent = max_negative_exponent
        self.use_siunitx = use_siunitx

    def _calculate_exponent(self, val: float, error: float) -> int:
        """Calculate the exponent based on value and error and the current exponent factor."""
        if self.exponent_factor <= 0 or (val == 0 and error == 0):
            return 0
        relevant_value = val if val != 0 else error
        exponent = int(
            np.floor(np.log10(abs(relevant_value)) / self.exponent_factor)
            * self.exponent_factor
        )
        if exponent > 0 and exponent < self.min_positive_exponent:
            exponent = 0
        if exponent < 0 and exponent > self.max_negative_exponent:
            exponent = 0
        return exponent

    def _calculate_precision(self, error: float, mean_exponent: int) -> int:
        """Calculate the precision based on error, error sigfigs, and mean exponent."""
        if error != 0:
            error_exponent = int(np.floor(np.log10(abs(error))))
            return self.err_sigfigs - error_exponent - 1
        return self.val_sigfigs - mean_exponent - 1

    def _format(self, mean: float, err: float) -> Tuple[str, str, int | None]:

        formatted_err = ""
        formatted_mean = ""
        formatted_exp = ""

        mean_exponent = self._calculate_exponent(mean, err)
        mantissa_mean = mean / (10.0**mean_exponent)
        mantissa_err = err / (10.0**mean_exponent)

        mantissa_precision = mean_exponent + self._calculate_precision(
            err, mean_exponent
        )
        if mantissa_precision > 0:
            formatted_mean = f"{mantissa_mean:.{mantissa_precision}f}"
            formatted_err = f"{mantissa_err:.{mantissa_precision}f}"
        else:
            formatted_mean = f"{int(mantissa_mean):d}"
            formatted_err = f"{int(mantissa_err):d}"
        if self.relative:
            relative_error = 100 * mantissa_err / mantissa_mean
            relative_precision = self._calculate_precision(relative_error, 0)
            formatted_err = (
                f"{relative_error:.{relative_precision}f}"
                if relative_precision > 0
                else f"{int(relative_error):d}"
            )

        if err == 0:
            formatted_err = "0"
        formatted_exp = mean_exponent if mean_exponent != 0 else None

        return formatted_mean, formatted_err, formatted_exp

    def latex_array(self, mean: np.ndarray, err: np.ndarray) -> np.ndarray:
        """
        Format the means and errors in the Err object as LaTeX strings.

        Args:
            mean (np.ndarray): the mean values.
            err (np.ndarray): the error values.
            delimiter (str): Delimiter for LaTeX math mode, default is '$'.
            use_siunitx (bool) : Whether to use the siunitx package

        Returns:
            np.ndarray: An array of formatted strings in LaTeX format.
        """

        @np.vectorize
        def format_to_latex(mean, err) -> str:
            formatted_mean, formatted_err, formatted_exp = self._format(mean, err)
            match self.use_siunitx, self.relative, formatted_exp is None:
                case True, True, True:
                    return rf"\SI{{{formatted_mean}}}{{\SI{{{formatted_err}}}{{\percent}}}}"
                case True, True, False:
                    return rf"\SI{{{formatted_mean}e{formatted_exp}}}{{\SI{{{formatted_err}}}{{\percent}}}}"
                case True, False, True:
                    return rf"\SI{{{formatted_mean}({formatted_err})}}{{}}"
                case True, False, False:
                    return (
                        rf"\SI{{{formatted_mean}({formatted_err})e{formatted_exp}}}{{}}"
                    )
                case False, True, True:
                    return rf"{formatted_mean} \pm {formatted_err}\%"
                case False, True, False:
                    return rf"({formatted_mean} \times 10^{{{formatted_exp}}}) \pm {formatted_err}\%"
                case False, False, True:
                    return rf"{formatted_mean} \pm {formatted_err}"
                case False, False, False:
                    return rf"({formatted_mean} \pm {formatted_err}) \times 10^{{{formatted_exp}}}"

        return format_to_latex(mean, err)

    def string_array(self, mean: np.ndarray, err: np.ndarray) -> np.ndarray:
        """
        Format the means and errors in the Err object as plain text strings.

        Args:
            mean (np.ndarray): the mean values.
            err (np.ndarray): the error values.

        Returns:
            np.ndarray: An array of formatted strings in plain text format.
        """

        @np.vectorize
        def format_to_string(mean, err) -> str:
            formatted_mean, formatted_err, formatted_exp = self._format(mean, err)
            match self.relative, formatted_exp is None:
                case True, True:
                    return f"{formatted_mean} ± {formatted_err}%"
                case True, False:
                    return f"({formatted_mean} ± {formatted_err}%) × 10^{formatted_exp}"
                case False, True:
                    return f"{formatted_mean} ± {formatted_err}"
                case False, False:
                    return f"({formatted_mean} ± {formatted_err}) × 10^{formatted_exp}"

        return format_to_string(mean, err)

    def __str__(self):
        return f"ErrFormat(err_sigfigs={self.err_sigfigs}, val_sigfigs={self.val_sigfigs}, relative={self.relative}, exponent_factor={self.exponent_factor}, min_positive_exponent={self.min_positive_exponent}, max_negative_exponent={self.max_negative_exponent})"


# Predefined ErrFormat instances for different formatting needs.
SCI_FORMAT = UErrFormatter(1, 5, False, 1, 3, -3)
SCI_FORMAT_REL = UErrFormatter(1, 5, True, 1, 3, -3)
ENG_FORMAT = UErrFormatter(2, 5, False, 3, 3, -3)
ENG_FORMAT_REL = UErrFormatter(2, 5, True, 3, 3, -3)
PLAIN_FORMAT = UErrFormatter(2, 5, False, 0, 0, 0)
PLAIN_FORMAT_REL = UErrFormatter(2, 5, True, 0, 0, 0)

# util/format.py
_default_formatter = SCI_FORMAT


def set_formatter(formatter: UErrFormatter):
    global _default_formatter
    _default_formatter = formatter


def get_formatter() -> UErrFormatter:
    """
    Returns a *copy* of the current global formatter.
    Changes to the returned object do not affect the global default;
    use set_formatter() to install a modified formatter.
    """
    return deepcopy(_default_formatter)
