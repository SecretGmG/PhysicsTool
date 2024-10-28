import numpy as np
import pandas as pd
import sympy
from numpy.typing import ArrayLike
from sympy import Symbol, Expr, Function, Eq
from typing import Optional, Dict, List, TextIO, Self

class Err:
    '''
    A class to handle measurements and their associated errors, providing various methods
    to compute and propagate errors in mathematical operations.

    Attributes:
        mean (np.ndarray): The mean values of the measurements.
        err (np.ndarray): The errors associated with the measurements.
    '''
    def __init__(self, mean: ArrayLike, err : ArrayLike = None):
        '''
        Initializes an Err object with mean values and associated errors.

        Parameters:
            mean (ArrayLike): The mean values of the measurements.
            err (Optional[ArrayLike]): The errors associated with the measurements. 
                If None, the errors are set to zero or extracted if 'mean' contains 
                Err instances.
        '''
        mean = np.asarray(mean)
        if err is None:
            if isinstance(mean.ravel()[0],Err):
                err = np.vectorize(lambda e: e.err)(mean)
                mean = np.vectorize(lambda e: e.mean)(mean)
            else:
                err = np.zeros_like(mean)
        
        mean, err = np.broadcast_arrays(mean, err)
        
        self.mean = np.atleast_1d(np.array(mean)).astype(np.float64)
        self.err = np.atleast_1d(np.array(err)).astype(np.float64)

    @classmethod
    def from_data(cls, data: ArrayLike, axis: Optional[int] = None) -> Self:
        '''
        Creates an Err instance from raw data by calculating the mean and standard error.

        Parameters:
            data (ArrayLike): Input data from which the mean and standard error will be calculated.
            axis (Optional[int]): The axis along which to calculate. If None, the entire data is used.

        Returns:
            Err: An Err object with calculated mean and standard error.
        '''
        mean = np.mean(data, axis)
        std_err = np.std(data, axis, ddof=1) / np.sqrt(np.size(data, axis))
        return cls(mean, std_err)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """
        Creates an Err instance from a pandas DataFrame with 'mean' and 'error' columns.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'mean' and 'error' columns.

        Returns:
            Err: An Err object with calculated mean and error.
        """
        if 'mean' not in df or 'error' not in df:
            raise ValueError("DataFrame must contain 'mean' and 'error' columns.")
        
        mean = df['mean'].values
        err = df['error'].values
        return cls(mean, err)


    def apply(self, foo: Function | Expr) -> Self:
        '''
        Applies a sympy function or expression to the current mean and error values.

        Parameters:
            foo (Function | Expr): The sympy function or expression to apply.

        Returns:
            Err: The resulting Err object after applying the function.
        '''
        x = Symbol('x')
        if isinstance(foo, sympy.FunctionClass):
            foo = foo(x)
        else:
            free_symbols = foo.free_symbols
            assert len(free_symbols) == 1, 'Cannot apply function with more than one argument. Use calc_err to compute such expressions.'
            x = list(free_symbols)[0]
        
        mean_result = sympy.lambdify(x, foo)(self.mean)
        err_result = sympy.lambdify(x, sympy.Abs(sympy.diff(foo, x)))(self.mean) * self.err
        return Err(mean_result, err_result)
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the upper and lower bounds for each measurement.

        Returns:
            tuple: A tuple (lower_bound, upper_bound), where both elements are numpy arrays.
        """
        lower_bound = self.mean - self.err
        upper_bound = self.mean + self.err
        return lower_bound, upper_bound
    def approx_eq(self, other: ArrayLike | Self, tolerance = 1.0) -> bool:
        '''
        Checks if the current values are approximately equal to another Err or array-like value.

        Parameters:
            other (ArrayLike | Err): The other value or Err object to compare against.
            tolerance (float): The tolerance level to account for the errors.

        Returns:
            bool: True if values are approximately equal within tolerance, False otherwise.
        '''
        if isinstance(other, Err):
            return np.allclose(self.mean, other.mean, atol=np.sqrt(self.err**2+other.err**2)*tolerance)
        else:
            return np.allclose(self.mean, other, atol=self.err*tolerance)

    def allclose(self, other : Self) -> bool:
        '''
        Checks if both the mean and error values are close to another Err object.

        Parameters:
            other (Err): The Err object to compare against.

        Returns:
            bool: True if both mean and error values are close, False otherwise.
        '''
        return np.allclose(self.mean, other.mean) and np.allclose(self.err, other.err)

    def average(self, axis=None) -> Self:
        '''
        Computes the average and propagates the error along a specified axis.

        Parameters:
            axis (Optional[int]): The axis along which to compute the average. If None, averages over all elements.

        Returns:
            Err: The resulting Err object after averaging.
        '''
        mean_avg = self.mean.mean(axis=axis)
        err_avg = np.sqrt(np.sum(self.err ** 2, axis=axis)) / np.size(self.err, axis=axis)
        return Err(mean_avg, err_avg)

    def weighted_average(self, axis=None) -> Self:
        '''
        Computes the weighted average and its associated error, following standard error propagation rules.

        Parameters:
            axis (Optional[int]): The axis along which to compute the weighted average. If None, averages over all elements.

        Returns:
            Err: The resulting Err object after weighted averaging.
        '''
        weights = 1 / (self.err ** 2)
        mean_weighted_avg = np.sum(self.mean * weights, axis=axis) / np.sum(weights, axis=axis)
        # source: https://www.physics.umd.edu/courses/Phys261/F06/ErrorPropagation.pdf
        err_weighted_avg = np.sqrt(1 / np.sum(weights, axis=axis))
        return Err(mean_weighted_avg, err_weighted_avg)

    def latex_array(self,
                    err_sigfigs: int = 2,
                    val_sigfigs: int = 5,
                    relative: bool = False,
                    expontent_factor: int = 3,
                    delimiter = '$'
                    ) -> ArrayLike:
        '''
        Generates a numpy string array with LaTeX-formatted scientific notation for all elements.

        Parameters:
            err_sigfigs (int): Number of significant figures for the error.
            relative (bool): Whether to format errors as relative percentages.
            delimiter (str): The delimiter used to enclose LaTeX expressions.

        Returns:
            ArrayLike: A numpy array containing LaTeX-formatted strings.
        '''
        formatted_mean, formatted_err, exponents = self._format_sci(err_sigfigs=err_sigfigs,val_sigfigs=val_sigfigs, relative=relative, expontent_factor=expontent_factor)
        latex_strings = np.vectorize(
            lambda m, e, exp: f'{delimiter}({m} \\pm {e}{'\\%' if relative else ''}) \\times 10^{{{exp}}}{delimiter}' if exp != 0 else f'{delimiter}{m} \\pm {e}{delimiter}{'\\%' if relative else ''}'
        )(formatted_mean, formatted_err, exponents)
        
        return latex_strings
    
    def string_array(self, 
                     err_sigfigs: int = 2, 
                     val_sigfigs: int = 5, 
                     relative: bool = False, 
                     expontent_factor: int = 3
                     ) -> ArrayLike:
        '''
        Returns a numpy string array formatted in scientific notation for all elements.

        Parameters:
            err_sigfigs (int): Number of significant figures for the error.
            val_sigfigs (int): Number of significant figures for the value.
            relative (bool): Whether to format errors as relative percentages.
            expontent_factor (int): The factor used to scale exponents (typically 3 for scientific notation).

        Returns:
            ArrayLike: A numpy array containing formatted strings.
        '''
        formatted_mean, formatted_err, exponents = self._format_sci(err_sigfigs, val_sigfigs, relative, expontent_factor)
        
        strings = np.vectorize(
            lambda m, e, exp: f'({m} ± {e}{'%' if relative else ''}) × 10^{exp}' if exp != 0 else f'{m} ± {e}{'%' if relative else ''}'
        )(formatted_mean, formatted_err, exponents)
        
        return strings

    def latex(self,
              err_sigfigs: int = 2,
              val_sigfigs: int = 5,
              relative: bool = False,
              expontent_factor: int = 3,
              delimiter: str = '$'
              ) -> str:
        '''
        Generates a LaTeX string for the error and mean values with customizable formatting.

        Parameters:
            err_sigfigs (int): Number of significant figures for the error.
            val_sigfigs (int): Number of significant figures for the value.
            relative (bool): Whether to format errors as relative percentages.
            delimiter (str): The delimiter used to enclose LaTeX expressions.

        Returns:
            str: A LaTeX-formatted string.
        '''
        latex_strings = np.ravel(self.latex_array(err_sigfigs=err_sigfigs, val_sigfigs=val_sigfigs, relative=relative, delimiter=delimiter, expontent_factor = expontent_factor))
        return r'\\'.join(latex_strings)

    def toString(self, err_sigfigs: int = 2, val_sigfigs: int = 5, relative: bool = False, expontent_factor: int = 3) -> str:
        '''
        Generates a string representation of the error and mean values with customizable formatting.

        Parameters:
            err_sigfigs (int): Number of significant figures for the error.
            val_sigfigs (int): Number of significant figures for the value.
            relative (bool): Whether to format errors as relative percentages.
            expontent_factor (int): The factor used to scale exponents (typically 3 for scientific notation).

        Returns:
            str: A formatted string.
        '''
        string_array = np.ravel(self.string_array(err_sigfigs=err_sigfigs, val_sigfigs=val_sigfigs, relative=relative, expontent_factor=expontent_factor))
        return '\n'.join(string_array)

    def __repr__(self):
        '''
        String representation of the Err object, showing its contents as formatted strings.
        '''
        return '\n'.join(np.ravel(self.string_array()))
    def _repr_latex_(self):
        '''
        Pretty LaTeX representation for Jupyter notebooks, using LaTeX formatting.
        '''
        return '$' + r'\\'.join(np.ravel(self.latex_array(delimiter = ''))) + '$'

    def _format_sci(self, err_sigfigs: int = 2, val_sigfigs: int = 5, relative: bool = False, expontent_factor: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Helper function to format mean and error values in scientific notation.

        Parameters:
            err_sigfigs (int): Number of significant figures for the error.
            val_sigfigs (int): Number of significant figures for the mean value.
            relative (bool): Whether to format errors as relative percentages.
            expontent_factor (int): The factor to use for scaling exponents.

        Returns:
            tuple: Formatted mean, error, and exponents as numpy arrays.
        '''
        
        formatted_means = np.empty_like(self.mean, dtype=object)
        formatted_errs = np.empty_like(self.err, dtype=object)
        exponents = np.empty_like(self.mean, dtype=int)

        # Flatten the arrays to handle both 1D and scalar inputs.
        mean_flat = np.ravel(self.mean)
        err_flat = np.ravel(self.err)

        for i, (val, error) in enumerate(zip(mean_flat, err_flat)):
            # Calculate the exponent based on the mean value
            mean_exponent = 0
            if expontent_factor <= 0:
                mean_exponent = 0
            elif val != 0:
                mean_exponent = int(np.floor(np.log10(abs(val)) / expontent_factor) * expontent_factor)
            elif error != 0:
                mean_exponent = int(np.floor(np.log10(abs(error)) / expontent_factor) * expontent_factor)
                
            # Determine the number of significant digits for the error
            if error != 0:
                error_exponent = int(np.floor(np.log10(abs(error))))
                precision = err_sigfigs - error_exponent - 1
            else:
                precision = val_sigfigs - mean_exponent - 1

            mantissa_mean = val / np.power(10.0,mean_exponent)
            mantissa_err = error / np.power(10.0,mean_exponent)
            
            matntissa_precision = mean_exponent+precision

            # Format mean and error based on their precision
            if matntissa_precision > 0:
                formatted_means[i] = f'{mantissa_mean:.{matntissa_precision}f}'
                formatted_errs[i] = f'{mantissa_err:.{matntissa_precision}f}'
            else:
                formatted_means[i] = f'{int(mantissa_mean):d}'
                formatted_errs[i] = f'{int(mantissa_err):d}'
            
            if relative and error != 0:
                relative_error = 100*mantissa_err/mantissa_mean
                relative_error_exponent = int(np.floor(np.log10(abs(relative_error))))
                relative_error_precision = err_sigfigs - relative_error_exponent - 1
                if relative_error_precision > 0:
                    formatted_errs[i] = f'{relative_error:.{relative_error_precision}f}'
                else:
                    formatted_errs[i] = f'{int(relative_error):d}'
                
            exponents[i] = mean_exponent

        # Handle edge case when error is 0
        mask = np.abs(self.err) = 0
        formatted_errs[mask] = '0'

        # Reshape back to the original shape of the input arrays
        return formatted_means.reshape(self.mean.shape), formatted_errs.reshape(self.err.shape), exponents.reshape(self.mean.shape)

    def __getitem__(self, indices):
        return Err(self.mean[indices], self.err[indices])
    
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

    def flatten(self):
        """
        Flattens the Err object into 1D. Returns an Err object where mean and err are 1D arrays.
        This is useful for handling multi-dimensional cases.
        """
        return Err(self.mean.flatten(), self.err.flatten())
    
    def __add__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(a+b, {a: self, b: other})

    def __sub__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(a-b, {a: self, b: other})

    def __mul__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(a*b, {a: self, b: other})

    def __pow__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(a**b, {a: self, b: other})

    def __truediv__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(a/b, {a: self, b: other})
    def __radd__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(b+a, {a: self, b: other})

    def __rsub__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(b-a, {a: self, b: other})

    def __rmul__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(b*a, {a: self, b: other})

    def __rpow__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(b**a, {a: self, b: other})

    def __rtruediv__(self, other):
        a, b = sympy.symbols('a,b')
        return calc_err(b/a, {a: self, b: other})

def derive_err(
    expr: Expr,
    values: Optional[List[Symbol]] = None,
    target_symbol: Symbol = Symbol('f'),
    err_prefix: str = 's',
    relative = False,
    do_display: bool = False,
    tex: Optional[TextIO] = None
) -> Expr:
    '''
    Derives the absolute Gaussian error from an expression.

    This function calculates the absolute Gaussian error for the given expression.

    Parameters:
        expr Expr: The expression to calculate the absolute Gaussian error for.
        values (Optional[List[Symbol]]): A list of values witch should be included in the error estimation. 
            If None, every free symbol is included. Defaults to None.
        target_symbol (Optional[Symbol]): The symbol of the expression to calculate the Gaussian error for. 
            Only used for logging. Defaults to 'f'.
        err_prefix (str): The string every error term is prefixed with. Defaults to 's'.
        relative (bool): all errors should be treated as relative
        do_display (bool) : wether to display the calculation and some of its intermediate result to the IPython output.
        tex (Optional[TextIO]): A file object to write LaTeX representation into. 
            Often it is useful to set this to sys.stdout. Defaults to None. 

    Returns:
        Expr: The expression for the absolute Gaussian error
    '''
    from .logging import log

    target_err_symbol = Symbol(f'{err_prefix}_{target_symbol}')
        
    target_err_symbol_squared = target_err_symbol ** 2

    if values is not None:
        free_args = list(set(expr.free_symbols) & set(values))
    else:
        free_args = list(expr.free_symbols)

    
    target_function = Function(target_symbol)(*free_args)
    temp_err_squared_expr = sympy.S.Zero
    temp_exprs, diff_exprs, diff_res_exprs = [], [], []

    for arg in free_args:
        t, s_arg = sympy.symbols(f'temp_{arg}, {err_prefix}_{arg}')
        if relative:
            temp_err_squared_expr += t ** 2 * s_arg**2 * arg**2
        else:
            temp_err_squared_expr += t ** 2 * s_arg**2
        d = sympy.diff(target_function, arg, evaluate=False)
        temp_exprs.append(t)
        diff_exprs.append(d)
        diff_res_exprs.append(d.subs(target_function, expr).doit())


    if relative:
        with sympy.evaluate(False):
            temp_err_squared_expr /= expr**2
    
    diff_err_squared_expr = temp_err_squared_expr.subs(
        list(zip(temp_exprs, diff_exprs)), evaluate=False)
    
    log(Eq(target_err_symbol_squared, diff_err_squared_expr,
            evaluate=False), do_display, tex)
        
    with sympy.evaluate(False):
        err_squared_expr = temp_err_squared_expr.subs(
            zip(temp_exprs, diff_res_exprs))
        
        log(Eq(target_err_symbol_squared, err_squared_expr), do_display, tex)
            
    err_expr = sympy.sqrt(err_squared_expr)
    
    log(Eq(target_err_symbol, err_expr, evaluate=False), do_display, tex)
    
    return err_expr


def calc_err(expr: sympy.Expr,
             values: Dict[sympy.Symbol, Err | ArrayLike]
             ) -> Err:
    '''
    Calculates the error of the given expression applied to the given values.
    
    Parameters:
        expr (Expr): The sympy expression for which to calculate the error.
        values (Dict[Symbol, Union[Err, ArrayLike]]): A dictionary mapping symbols to Err instances or array-like values.
        
    Returns:
        Err: The computed error as an Err instance.
    '''
    err_prefix = 'temporary_error_prefix'

    mean_values = {}
    err_values = {}
    
    for key, val in values.items():
        if not isinstance(val, Err):
            values[key] = Err(val)
            
    for key, val in values.items():
        mean_values[key] = val.mean
        err_values[Symbol(f'{err_prefix}_{key}')] = val.err

    all_values = mean_values | err_values
    err_expr = derive_err(expr, mean_values.keys(), err_prefix=err_prefix)

    mean_func = sympy.lambdify(all_values.keys(), expr)
    err_func = sympy.lambdify(all_values.keys(), err_expr)

    return Err(mean_func(*all_values.values()), err_func(*all_values.values()))