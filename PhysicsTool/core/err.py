import numpy as np
import sympy
from numpy.typing import ArrayLike
from typing import Iterable, Optional, Dict, List, TextIO

a, b = sympy.symbols('a,b')

class Err:
    '''
    A class to handle measurements and their associated errors, providing various methods
    to compute and propagate errors in mathematical operations.

    Attributes:
        mean (np.ndarray): The mean values of the measurements.
        err (np.ndarray): The errors associated with the measurements.
    '''
    
    # Ensures Err takes precedence over NumPy arrays, leading to the correct ufunc behavior
    __array_priority__ = 1000

    def __init__(self, mean: ArrayLike, err : Optional[ArrayLike] = None, format : Optional['ErrFormat'] = None, relative = False): # type: ignore
        '''
        Initializes an Err object with mean values and associated errors.

        Parameters:
            mean (ArrayLike): The mean values of the measurements.
            err (Optional[ArrayLike]): The errors associated with the measurements. 
                If None, the errors are set to zero or extracted if 'mean' contains 
                Err instances.
        '''
        if format is None:
            from PhysicsTool.core.format import SCI_FORMAT, SCI_FORMAT_REL
            if relative:
                format = SCI_FORMAT_REL
            else:
                format = SCI_FORMAT
        
        self.format = format
        
        if isinstance(mean, Err):
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
    
    
    def with_format(self, format: 'ErrFormat') -> 'Err': # type: ignore
        '''
        Returns a copy of the Err object with a new format.

        Parameters:
            format (ErrFormat): The new format to apply to the Err object.

        Returns:
            Err: A new Err object with the specified format.
        '''
        return Err(self.mean, self.err, format)
    
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Returns the upper and lower bounds for each measurement.

        Returns:
            tuple: A tuple (lower_bound, upper_bound), where both elements are numpy arrays.
        '''
        lower_bound = self.mean - self.err
        upper_bound = self.mean + self.err
        return lower_bound, upper_bound
    
    def approx_eq(self, other: ArrayLike | 'Err', tolerance: float = 1.0) -> bool:
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

    def allclose(self, other : 'Err') -> bool:
        '''
        Checks if both the mean and error values are close to another Err object.

        Parameters:
            other (Err): The Err object to compare against.

        Returns:
            bool: True if both mean and error values are close, False otherwise.
        '''
        return np.allclose(self.mean, other.mean) and np.allclose(self.err, other.err)

    def average(self, axis: Optional[int] = None) -> 'Err':
        '''
        Computes the average and propagates the error along a specified axis.

        Parameters:
            axis (Optional[int]): The axis along which to compute the average. If None, averages over all elements.

        Returns:
            Err: The resulting Err object after averaging.
        '''
        mean_avg = self.mean.mean(axis=axis)
        err_avg = np.sqrt(np.sum(self.err ** 2, axis=axis)) / np.size(self.err, axis=axis)
        return Err(mean_avg, err_avg, format = self.format)

    def weighted_average(self, axis: Optional[int] = None) -> 'Err':
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
        return Err(mean_weighted_avg, err_weighted_avg, format = self.format)

    def latex(self, use_siunitx : bool = True) -> str:
        '''
        Generates a LaTeX string for the error and mean values with customizable formatting.

        Parameters:
            delimiter (str): The delimiter used to enclose LaTeX expressions.
            use_siunitx (bool) : Wether to use the siunitx package

        Returns:
            str: A LaTeX-formatted string.
        '''
        return '\\\\\n'.join(np.ravel(self.format.latex_array(self, use_siunitx)))

    def flatten(self):
        '''
        Flattens the Err object into 1D. Returns an Err object where mean and err are 1D arrays.
        This is useful for handling multi-dimensional cases.
        '''
        return Err(self.mean.flatten(), self.err.flatten())
    
    def __str__(self) -> str:
        '''
        Generates a string representation of the error and mean values with customizable formatting.
        
        Returns:
            str: A formatted string.
        '''
        return '\n'.join(np.ravel(self.format.string_array(self)))

    def __repr__(self):
        '''
        String representation of the Err object, showing its contents as formatted strings.
        '''
        return self.__str__()
    def _repr_latex_(self):
        '''
        Pretty LaTeX representation for Jupyter notebooks, using LaTeX formatting.
        '''
        return '$' + self.latex(use_siunitx=False) + '$'

    def __getitem__(self, indices):
        return Err(self.mean[indices], self.err[indices], format=self.format)
    
    def __len__(self):
        '''
        Return the length of the first dimension of the Err object.
        If the array is multi-dimensional, this returns the size of the first axis.
        '''
        return self.mean.shape[0]
    
    def __iter__(self):
        '''
        Allows the Err object to be treated as an iterable (iterating over its elements).
        This makes it compatible with pandas when assigned to a DataFrame column.
        If the Err object is multi-dimensional, this flattens the object, yielding
        Err objects for each element.
        '''
        # Flatten along the first dimension and iterate over that
        for i in range(self.mean.shape[0]):
            yield self[i]  # Return an Err object for each "row" or slice
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        Implements the array ufunc protocol for Err objects, allowing for element-wise operations.
        
        Parameters:
            ufunc (numpy.ufunc): The numpy universal function to apply.
            method (str): The method to apply, typically '__call__'.
            inputs (Tuple): The input values to apply the ufunc to.
            kwargs (Dict): Additional keyword arguments.
            
        Returns:
            Err: The resulting Err object after applying the ufunc.
        '''
        
        assert method == '__call__', 'Only __call__ method is supported'
        
        sympy_func_name = ufunc.__name__.replace('arc', 'a')
        
        assert hasattr(sympy, sympy_func_name),f'No equivalent sympy function for {ufunc.__name__}'
        sympy_ufunc = getattr(sympy, sympy_func_name)
        assert isinstance(sympy_ufunc, sympy.FunctionClass), f'No equivalent sympy function for {ufunc.__name__}'
        
        nr_symbols = sympy.numbered_symbols()
        symbols = [next(nr_symbols) for _ in range(len(inputs)+len(kwargs))]
        try:
            out = calc_err(sympy_ufunc(*symbols), dict(zip(symbols, inputs)))
        except:
            raise ValueError(f'Error in applying {ufunc.__name__} to Err objects')        
        return out
    
    def __eq__(self, other):
        if isinstance(other, Err):
            return np.array_equal(self.mean, other.mean) and np.array_equal(self.err, other.err)
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
        return Err(-self.mean, self.err, format=self.format)

    def __abs__(self):
        return Err(np.abs(self.mean), self.err, format=self.format)
    
    def __add__(self, other):
        return calc_err(a+b, {a: self, b: other}).with_format(_combine_format(self, other))

    def __sub__(self, other):
        return calc_err(a-b, {a: self, b: other}).with_format(_combine_format(self, other))

    def __mul__(self, other):
        return calc_err(a*b, {a: self, b: other}).with_format(_combine_format(self, other))

    def __pow__(self, other):
        return calc_err(a**b, {a: self, b: other}).with_format(_combine_format(self, other))

    def __truediv__(self, other):
        return calc_err(a/b, {a: self, b: other}).with_format(_combine_format(self, other))
    
    def __radd__(self, other):
        return calc_err(b+a, {a: self, b: other}).with_format(_combine_format(self, other))

    def __rsub__(self, other):
        return calc_err(b-a, {a: self, b: other}).with_format(_combine_format(self, other))

    def __rmul__(self, other):
        return calc_err(b*a, {a: self, b: other}).with_format(_combine_format(self, other))

    def __rpow__(self, other):
        return calc_err(b**a, {a: self, b: other}).with_format(_combine_format(self, other))

    def __rtruediv__(self, other):
        return calc_err(b/a, {a: self, b: other}).with_format(_combine_format(self, other))

def concatenate(errs : Iterable['Err'], axis = 0) -> 'Err':
    '''
    Concatenates an Iterable of Err objects into a single Err object by combining them.
    
    Parameters:
        errs (Iterable[Err]): The Err objects to concatenate.
        axis (int): The axis along which to concatenate the Err objects.
        
    Returns:
        Err: The resulting Err object after concatenation.
    
    '''
    mean = np.concatenate([err.mean for err in errs], axis=axis)
    err = np.concatenate([err.err for err in errs], axis=axis)
    return Err(mean, err, format = errs[0].format)

def from_data(data : ArrayLike, axis: Optional[int] = None, format: Optional['ErrFormat'] = None) -> Err: # type: ignore
    '''
    Creates an Err instance from raw data by calculating the mean and standard error.
    Parameters:
        data (ArrayLike): Input data from which the mean and standard error will be calculated.
        axis (Optional[int]): The axis along which to calculate. If None, the entire data is used.
    Returns:
        Err: An Err object with calculated mean and standard error.
    '''
    mean = np.nanmean(data, axis)
    std_err = np.nanstd(data, axis, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(data), axis))
    return Err(mean, std_err, format = format)


def _combine_format(a, b):
    if isinstance(a, Err) and isinstance(b, Err):
        return a.format.combine(b.format)
    if isinstance(a, Err):
        return a.format
    if isinstance(b, Err):
        return b.format
    from PhysicsTool.core.format import SCI_FORMAT
    return SCI_FORMAT

def derive_err(
    expr: sympy.Expr,
    values: Optional[List[sympy.Symbol]] = None,
    target_symbol: sympy.Symbol = sympy.Symbol('f'),
    err_prefix: str = 's',
    relative: bool = False,
    do_display: bool = False,
    tex: Optional[TextIO] = None
) -> sympy.Expr:
    '''
    Derives the absolute or relative Gaussian error from an expression, based on propagation rules.

    This function calculates the Gaussian error for the given expression by computing
    the error contribution from each variable. For each variable, it computes the partial derivative 
    with respect to that variable, squares it, and multiplies it by the square of the associated error.
    The results are summed and then square-rooted to yield the total error expression.

    If `relative` is set to `True`, each term is additionally multiplied by the square of the variable's 
    value, giving the relative error propagation form. This is useful for proportional errors 
    or when error terms are intended as percentages of the mean values.

    Parameters:
        expr (Expr): The sympy expression to calculate the Gaussian error for.
        values (Optional[List[Symbol]]): A list of symbols to include in error estimation. 
            If None, all free symbols in `expr` are used. Defaults to None.
        target_symbol (Symbol): Symbol representing the final expression for logging purposes.
            Defaults to 'f'.
        err_prefix (str): Prefix for the symbol names representing each variable's error term.
            Defaults to 's'.
        relative (bool): Whether to treat all errors as relative to the values they measure.
            When True, each error term is scaled by the variable's value in `expr`.
            Defaults to False.
        do_display (bool): Whether to display the calculation and intermediate results in
            IPython output. Useful for debugging or documentation. Defaults to False.
        tex (Optional[TextIO]): File object to write LaTeX representation into, often
            set to `sys.stdout` for console output. Defaults to None. 

    Returns:
        Expr: The sympy expression for the Gaussian error in `expr`.
    
    Example:
        To calculate the error of an expression with relative errors:

            x, y = sympy.symbols('x y')
            expr = x * y
            error_expr = derive_err(expr, relative=True)
    '''

    from .logging import log

    target_err_symbol = sympy.Symbol(f'{err_prefix}_{target_symbol}')
        
    target_err_symbol_squared = target_err_symbol ** 2

    if values is not None:
        free_args = list(set(expr.free_symbols) & set(values))
    else:
        free_args = list(expr.free_symbols)

    
    target_function = sympy.Function(target_symbol)(*free_args)
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
    
    log(sympy.Eq(target_err_symbol_squared, diff_err_squared_expr,
            evaluate=False), do_display, tex)
        
    with sympy.evaluate(False):
        err_squared_expr = temp_err_squared_expr.subs(
            zip(temp_exprs, diff_res_exprs))
        
        log(sympy.Eq(target_err_symbol_squared, err_squared_expr), do_display, tex)
            
    err_expr = sympy.sqrt(err_squared_expr)
    
    log(sympy.Eq(target_err_symbol, err_expr, evaluate=False), do_display, tex)
    
    return err_expr


def calc_err(expr: sympy.Expr,
             values: Dict[sympy.Symbol, Err | ArrayLike],
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
    
    _check_for_missing_symbols(expr, values)
    
    #cast all values to errors
    values = {key: Err(val) for key, val in values.items()}
    
    mean_values = {key : val.mean for key, val in values.items()}
    err_values = {sympy.Symbol(f'{err_prefix}_{key}'): val.err for key, val in values.items()}

    all_values = mean_values | err_values
    
    err_expr = derive_err(expr, mean_values.keys(), err_prefix=err_prefix)

    mean_func = sympy.lambdify(all_values.keys(), expr)
    err_func = sympy.lambdify(all_values.keys(), err_expr)

    return Err(mean_func(*all_values.values()), err_func(*all_values.values()))

def _check_for_missing_symbols(expr, values):
    missing_symbols = [symbol for symbol in expr.free_symbols if symbol not in values]
    if missing_symbols:
        missing_symbols_string = ', '.join([str(s) for s in missing_symbols])
        raise ValueError(f'Missing value for required symbols: {missing_symbols_string}')