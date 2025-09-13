import numpy as np
import sympy
from numpy.typing import ArrayLike
from typing import Iterable, Optional, Dict
from u_err_format import get_formatter, UErrFormatter
from error_prop import derive_err

a, b = sympy.symbols("a,b")


class UErr:
    """
    A class to handle measurements and their associated errors, providing various methods
    to compute and propagate errors in mathematical operations.

    Attributes:
        mean (np.ndarray): The mean values of the measurements.
        err (np.ndarray): The errors associated with the measurements.
    """

    mean: np.ndarray
    err: np.ndarray

    # Ensures Err takes precedence over NumPy arrays, leading to the correct ufunc behavior
    __array_priority__ = 1000

    def __init__(
        self, mean: ArrayLike | "UErr", err: Optional[ArrayLike] = None, relative=False
    ):
        """
        Initializes an Err object with mean values and associated errors.

        Parameters:
            mean (ArrayLike): The mean values of the measurements.
            err (Optional[ArrayLike]): The errors associated with the measurements.
                If None, the errors are set to zero or extracted if 'mean' contains
                Err instances.
        """
        if isinstance(mean, UErr):
            err = mean.err
            mean = mean.mean

        mean = np.array(mean)

        if err is None:
            err = np.zeros_like(mean)

        err = np.array(err)

        if relative:
            err = mean * err

        mean, err = np.broadcast_arrays(mean, err)

        self.mean = np.atleast_1d(mean).astype(np.float64)
        self.err = np.atleast_1d(err).astype(np.float64)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the upper and lower bounds for each measurement.

        Returns:
            tuple: A tuple (lower_bound, upper_bound), where both elements are numpy arrays.
        """
        lower_bound = self.mean - self.err
        upper_bound = self.mean + self.err
        return lower_bound, upper_bound

    def approx_eq(self, other: ArrayLike | "UErr", tolerance: float = 1.0) -> bool:
        """
        Checks if the current values are approximately equal to another Err or array-like value.

        Parameters:
            other (ArrayLike | Err): The other value or Err object to compare against.
            tolerance (float): The tolerance level to account for the errors.
                A tolerance of 1 checks if the difference of the value lies within one sigma.

        Returns:
            bool: True if values are approximately equal within tolerance, False otherwise.
        """

        other_uerr = UErr(other)
        combined_err = np.sqrt(self.err**2 + other_uerr.err**2) + np.finfo(float).eps

        return np.allclose(
            self.mean / combined_err,
            other_uerr.mean / combined_err,
            atol=tolerance,
        )

    def allclose(self, other: "UErr") -> bool:
        """
        Checks if both the mean and error values are close to another Err object.

        Parameters:
            other (Err): The Err object to compare against.

        Returns:
            bool: True if both mean and error values are close, False otherwise.
        """
        return np.allclose(self.mean, other.mean) and np.allclose(self.err, other.err)

    def average(self, axis: Optional[int] = None) -> "UErr":
        """
        Computes the average and propagates the error along a specified axis.

        Parameters:
            axis (Optional[int]): The axis along which to compute the average. If None, averages over all elements.

        Returns:
            Err: The resulting Err object after averaging.
        """
        mean_avg = self.mean.mean(axis=axis)
        err_avg = np.sqrt(np.sum(self.err**2, axis=axis)) / np.size(self.err, axis=axis)
        return UErr(mean_avg, err_avg)

    def weighted_average(self, axis: Optional[int] = None) -> "UErr":
        """
        Computes the weighted average and its associated error, following standard error propagation rules.

        Parameters:
            axis (Optional[int]): The axis along which to compute the weighted average. If None, averages over all elements.

        Returns:
            Err: The resulting Err object after weighted averaging.
        """
        weights = 1 / (self.err**2)
        mean_weighted_avg = np.sum(self.mean * weights, axis=axis) / np.sum(
            weights, axis=axis
        )
        # source: https://www.physics.umd.edu/courses/Phys261/F06/ErrorPropagation.pdf
        err_weighted_avg = np.sqrt(1 / np.sum(weights, axis=axis))
        return UErr(mean_weighted_avg, err_weighted_avg)

    def latex(self, formatter: UErrFormatter | None = None) -> str:
        """
        Generates a LaTeX string for the error and mean values with customizable formatting.

        Parameters:
            delimiter (str): The delimiter used to enclose LaTeX expressions.
            use_siunitx (bool) : Wether to use the siunitx package

        Returns:
            str: A LaTeX-formatted string.
        """
        if formatter is None:
            formatter = get_formatter()
        return "\\\\\n".join(np.ravel(formatter.latex_array(self.mean, self.err)))

    def flatten(self):
        """
        Flattens the Err object into 1D. Returns an Err object where mean and err are 1D arrays.
        This is useful for handling multi-dimensional cases.
        """
        return UErr(self.mean.flatten(), self.err.flatten())

    def __str__(self) -> str:
        """
        Generates a string representation of the error and mean values with customizable formatting.

        Returns:
            str: A formatted string.
        """
        return "\n".join(np.ravel(get_formatter().string_array(self.mean, self.err)))

    def __repr__(self):
        """
        String representation of the Err object, showing its contents as formatted strings.
        """
        return self.__str__()

    def _repr_latex_(self):
        """
        Pretty LaTeX representation for Jupyter notebooks, using LaTeX formatting.
        """
        formatter = get_formatter()
        formatter.use_siunitx = False
        return "$" + self.latex(formatter) + "$"

    def __getitem__(self, indices):
        return UErr(self.mean[indices], self.err[indices])

    def __len__(self):
        """
        Return the length of the first dimension of the Err object.
        If the array is multi-dimensional, this returns the size of the first axis.
        """
        return self.mean.shape[0]

    def __iter__(self):
        """
        Allows the Err object to be treated as an iterable (iterating over its elements).
        This makes it compatible with pandas when assigned to a DataFrame column.
        If the Err object is multi-dimensional, this flattens the object, yielding
        Err objects for each element.
        """
        # Flatten along the first dimension and iterate over that
        for i in range(self.mean.shape[0]):
            yield self[i]  # Return an Err object for each "row" or slice

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Implements the array ufunc protocol for Err objects, allowing for element-wise operations.

        Parameters:
            ufunc (numpy.ufunc): The numpy universal function to apply.
            method (str): The method to apply, typically '__call__'.
            inputs (Tuple): The input values to apply the ufunc to.
            kwargs (Dict): Additional keyword arguments.

        Returns:
            Err: The resulting Err object after applying the ufunc.
        """

        assert method == "__call__", "Only __call__ method is supported"

        sympy_func_name = ufunc.__name__.replace("arc", "a")

        assert hasattr(
            sympy, sympy_func_name
        ), f"No equivalent sympy function for {ufunc.__name__}"
        sympy_ufunc = getattr(sympy, sympy_func_name)
        assert isinstance(
            sympy_ufunc, sympy.FunctionClass
        ), f"No equivalent sympy function for {ufunc.__name__}"

        nr_symbols = sympy.numbered_symbols()
        symbols = [next(nr_symbols) for _ in range(len(inputs) + len(kwargs))]
        try:
            out = calc_err(sympy_ufunc(*symbols), dict(zip(symbols, inputs)))
        except Exception:
            raise ValueError(f"Error in applying {ufunc.__name__} to Err objects")
        return out

    def __eq__(self, other):
        if isinstance(other, UErr):
            return np.array_equal(self.mean, other.mean) and np.array_equal(
                self.err, other.err
            )
        return False  # Err objects are not equal to non-Err types

    def __gt__(self, other):
        return self.mean > other.mean

    def __lt__(self, other):
        return self.mean < other.mean

    def __ge__(self, other):
        return self.mean >= other.mean

    def __le__(self, other):
        return self.mean <= other.mean

    def __neg__(self):
        return UErr(-self.mean, self.err)

    def __abs__(self):
        return UErr(np.abs(self.mean), self.err)

    def __add__(self, other):
        return calc_err(a + b, {a: self, b: other})

    def __sub__(self, other):
        return calc_err(a - b, {a: self, b: other})

    def __mul__(self, other):
        return calc_err(a * b, {a: self, b: other})

    def __pow__(self, other):
        return calc_err(a**b, {a: self, b: other})

    def __truediv__(self, other):
        return calc_err(a / b, {a: self, b: other})

    def __radd__(self, other):
        return calc_err(b + a, {a: self, b: other})

    def __rsub__(self, other):
        return calc_err(b - a, {a: self, b: other})

    def __rmul__(self, other):
        return calc_err(b * a, {a: self, b: other})

    def __rpow__(self, other):
        return calc_err(b**a, {a: self, b: other})

    def __rtruediv__(self, other):
        return calc_err(b / a, {a: self, b: other})


def concatenate(errs: Iterable["UErr"], axis=0) -> "UErr":
    """
    Concatenates an Iterable of Err objects into a single Err object by combining them.

    Parameters:
        errs (Iterable[Err]): The Err objects to concatenate.
        axis (int): The axis along which to concatenate the Err objects.

    Returns:
        Err: The resulting Err object after concatenation.

    """
    mean = np.concatenate([err.mean for err in errs], axis=axis)
    err = np.concatenate([err.err for err in errs], axis=axis)
    return UErr(mean, err)


def from_data(data: ArrayLike, axis: Optional[int] = None) -> UErr:
    """
    Creates an Err instance from raw data by calculating the mean and standard error.
    Parameters:
        data (ArrayLike): Input data from which the mean and standard error will be calculated.
        axis (Optional[int]): The axis along which to calculate. If None, the entire data is used.
    Returns:
        Err: An Err object with calculated mean and standard error.
    """
    mean = np.nanmean(np.asarray(data), axis)
    std_err = np.nanstd(np.asarray(data), axis, ddof=1) / np.sqrt(
        np.count_nonzero(~np.isnan(data), axis)
    )
    return UErr(mean, std_err)


def calc_err(
    expr: sympy.Expr,
    values: Dict[sympy.Symbol, UErr | ArrayLike],
) -> UErr:
    """
    Calculates the error of the given expression applied to the given values.

    Parameters:
        expr (Expr): The sympy expression for which to calculate the error.
        values (Dict[Symbol, Union[Err, ArrayLike]]): A dictionary mapping symbols to Err instances or array-like values.

    Returns:
        Err: The computed error as an Err instance.
    """

    err_prefix = "temporary_error_prefix"

    _check_for_missing_symbols(expr, values)

    # cast all values to errors
    uerr_values: dict[sympy.Symbol, UErr] = {
        key: UErr(val) for key, val in values.items()
    }

    mean_values: dict[sympy.Symbol, np.ndarray] = {
        key: val.mean for key, val in uerr_values.items()
    }
    err_values = {
        sympy.Symbol(f"{err_prefix}_{key}"): val.err for key, val in uerr_values.items()
    }

    all_values = mean_values | err_values

    err_expr = derive_err(expr, list(mean_values.keys()), err_prefix=err_prefix)

    mean_func = sympy.lambdify(all_values.keys(), expr)
    err_func = sympy.lambdify(all_values.keys(), err_expr)

    return UErr(mean_func(*all_values.values()), err_func(*all_values.values()))


def _check_for_missing_symbols(expr, values):
    missing_symbols = [symbol for symbol in expr.free_symbols if symbol not in values]
    if missing_symbols:
        missing_symbols_string = ", ".join([str(s) for s in missing_symbols])
        raise ValueError(
            f"Missing value for required symbols: {missing_symbols_string}"
        )
