import numpy as np
import sympy
from numpy.typing import ArrayLike
from sympy import Symbol, Expr, Function, Eq
from typing import Optional, Dict, List, TextIO, Self

class Err:
    """
    A class to handle measurements and their associated errors.
    
    Attributes:
        mean (np.ndarray): The mean values of the measurements.
        err (np.ndarray): The errors associated with the measurements.
    """
    def __init__(self, mean: ArrayLike, err : ArrayLike = None):
        """
        Initializes an instance of the Err class.
        
        Parameters:
            mean (ArrayLike): The mean values of the measurements.
            err (Optional[ArrayLike]): The errors associated with the measurements. If None, initializes with zero errors.
        """
        mean = np.asarray(mean)
        if err is None:
            if isinstance(mean.ravel()[0],Err):
                err = np.vectorize(lambda e: e.err)(mean)
                mean = np.vectorize(lambda e: e.mean)(mean)
            else:
                err = np.zeros_like(mean)
        
        mean, err = np.broadcast_arrays(mean, err)
        
        self.mean = np.array(mean)
        self.err = np.array(err)

    @classmethod
    def from_data(cls, data: ArrayLike, axis: Optional[int] = None) -> Self:
        """
        Creates an Err instance from data by calculating the mean and standard error of the mean.
        
        Parameters:
            data (ArrayLike): The input data.
            axis (Optional[int]): The axis along which to calculate the mean and standard error.
        
        Returns:
            Err: The created Err instance.
        """
        mean = np.mean(data, axis)
        std_err = np.std(data, axis, ddof=1) / np.sqrt(np.size(data, axis))
        return cls(mean, std_err)

    def apply(self, foo: Function | Expr) -> Self:
        """
        Applies the given function or expression to self.
        
        Parameters:
            foo (Function | Expr): The function or expression to apply.
        
        Returns:
            Err: The resulting Err instance.
        """
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
    
    def approx_eq(self, other: ArrayLike | Self) -> bool:
        """
        Returns True if both values lie within each other's error bounds.
        
        Parameters:
            other (ArrayLike | Err): The other value to compare with.
        
        Returns:
            bool: True if the values are approximately equal, False otherwise.
        """
        if isinstance(other, Err):
            return np.allclose(self.mean, other.mean, atol=np.sqrt(self.err**2+other.err**2))
        else:
            return np.allclose(self.mean, other, atol=self.err)

    def allclose(self, other : Self) -> bool:
        """
        Returns True if both values' means and errors are close to each other.
        
        Parameters:
            other (Err): The other Err instance to compare with.
        
        Returns:
            bool: True if the values are all close, False otherwise.
        """
        return np.allclose(self.mean, other.mean) and np.allclose(self.err, other.err)

    def average(self, axis=None) -> Self:
        """
        Computes the average and its propagated error along a given axis.
        
        Parameters:
            axis (Optional[int]): The axis along which to compute the average.
        
        Returns:
            Err: The resulting Err instance.
        """
        mean_avg = self.mean.mean(axis=axis)
        err_avg = np.sqrt(np.sum(self.err ** 2, axis=axis)) / np.size(self.err, axis=axis)
        return Err(mean_avg, err_avg)

    def weighted_average(self, axis=None) -> Self:
        """
        Calculate the weighted average and combined error along a given axis.
        this method is explained here:
        https://www.physics.umd.edu/courses/Phys261/F06/ErrorPropagation.pdf
        
        Parameters:
            axis (Optional[int]): The axis along which to compute the weighted average.
        
        Returns:
            Err: The resulting Err instance.
        """
        weights = 1 / (self.err ** 2)
        mean_weighted_avg = np.sum(self.mean * weights, axis=axis) / np.sum(weights, axis=axis)
        err_weighted_avg = np.sqrt(1 / np.sum(weights, axis=axis))
        return Err(mean_weighted_avg, err_weighted_avg)

    def _latex_cores(self,
              sigfigs: int = 2,
              min_precision: int = 0,
              max_precision: int = 16,
              relative: bool = False
        ):
        mean_str, err_str = self.formatted(
            sigfigs, min_precision, max_precision, relative)
        return mean_str + np.choose(err_str == '0', ['\pm' + err_str + (r'\%' if relative else ''), ''])

    def latex_array(self,
              sigfigs: int = 2,
              min_precision: int = 0,
              max_precision: int = 16,
              relative: bool = False
              ) -> ArrayLike:
        """
        Returns a numpy string array with a LaTeX expression for every element.
        
        Parameters:
            sigfigs (int): Significant figures.
            min_precision (int): Minimum precision.
            max_precision (int): Maximum precision.
            relative (bool): Whether to format errors as relative percentages.
        
        Returns:
            ArrayLike: LaTeX formatted array.
        """
        return '$' + self._latex_cores(sigfigs, min_precision, max_precision, relative) + '$'

    def _repr_latex_(self):
        return '$'+r',  \\'.join(np.ravel([self._latex_cores()])) +'$'

    def formatted(self,
                   sigfigs: int = 2,
                   min_precision: int = 0,
                   max_precision: int = 16,
                   relative: bool = False,
                   ) -> ArrayLike:
        """
        Formats the mean and error values.
        
        Parameters:
            sigfigs (int): Significant figures.
            min_precision (int): Minimum precision.
            max_precision (int): Maximum precision.
            relative (bool): Whether to format errors as relative percentages.
        
        Returns:
            tuple: Formatted mean and error strings.
        """
        temp_err = self.err.copy()
        if relative:
            temp_err *= 100/self.mean

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            zeroformatted = _fmt_to_order(self.mean, np.minimum(_get_order(
                self.mean),0) - sigfigs, min_precision, max_precision)
            normalformatted = _fmt_to_order(self.mean, _get_order(
                self.err) - sigfigs, min_precision, max_precision)
            normalformatted_err = _fmt_to_order(temp_err, _get_order(
                temp_err) - sigfigs, min_precision, max_precision)

        mask = np.isclose(self.err, 0)

        mean_str = normalformatted
        err_str = normalformatted_err

        mean_str[mask] = zeroformatted[mask]
        err_str[mask] = '0'

        return (mean_str, err_str)

    def formatted_mean(self,
                   sigfigs: int = 2,
                   min_precision: int = 0,
                   max_precision: int = 16,
                   relative: bool = False,
                   ) -> ArrayLike:
        """
        Formats the mean values.
        
        Parameters:
            sigfigs (int): Significant figures.
            min_precision (int): Minimum precision.
            max_precision (int): Maximum precision.
            relative (bool): Whether to format errors as relative percentages.
        
        Returns:
            ArrayLike: Formatted mean strings.
        """
        return self.formatted(sigfigs, min_precision, max_precision, relative)[0]
    
    def formatted_err(self,
                   sigfigs: int = 2,
                   min_precision: int = 0,
                   max_precision: int = 16,
                   relative: bool = False,
                   ) -> ArrayLike:
        """
        Formats the error values.
        
        Parameters:
            sigfigs (int): Significant figures.
            min_precision (int): Minimum precision.
            max_precision (int): Maximum precision.
            relative (bool): Whether to format errors as relative percentages.
        
        Returns:
            ArrayLike: Formatted error strings.
        """
        return self.formatted(sigfigs, min_precision, max_precision, relative)[1]
    
    def __getitem__(self, indices):
        return Err(self.mean[indices], self.err[indices])
    
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

    def __repr__(self):
        mean_str, err_str = self.formatted()
        return f'{mean_str} ± {err_str}'

@np.vectorize(otypes = [int])
def _get_order(num: float) -> int:
    if num == 0:
        return -40 # corresponds to float min value of ~ 1e-39
    return int(np.floor(np.log10(abs(num)))) + 1


@np.vectorize(otypes = ['O'])
def _fmt_to_order(num: float, order: int, min_precision: int, max_precision: int) -> str:
    """
    Formats the number to the specified order of magnitude with given precision constraints.
    
    Parameters:
        num (float): The input number.
        order (int): The order of magnitude to format to.
        min_precision (int): The minimum precision of the formatted number.
        max_precision (int): The maximum precision of the formatted number.
        
    Returns:
        str: The formatted number as a string.
    """
    precision = np.clip(-order, min_precision, max_precision)
    return f"{round(num,-order):.{precision}f}"

def derive_err(
    expr: Expr,
    values: Optional[List[Symbol]] = None,
    target_symbol: Symbol = Symbol('f'),
    err_prefix: str = 's',
    do_display: bool = False,
    tex: Optional[TextIO] = None
) -> Expr:
    """
    Derives the absolute Gaussian error from an expression.

    This function calculates the absolute Gaussian error for the given expression.

    Parameters:
        expr Expr: The expression to calculate the absolute Gaussian error for.
        values (Optional[List[Symbol]]): A list of values witch should be included in the error estimation. 
            If None, every free symbol is included. Defaults to None.
        target_symbol (Optional[Symbol]): The symbol of the expression to calculate the Gaussian error for. 
            Only used for logging. Defaults to 'f'.
        err_prefix (str): The string every error term is prefixed with. Defaults to 's'.
        do_display (bool) : wether to display the calculation and some of its intermediate result to the IPython output.
        tex (Optional[TextIO]): A file object to write LaTeX representation into. 
            Often it is useful to set this to sys.stdout. Defaults to None. 

    Returns:
        Expr: The expression for the absolute Gaussian error
    """
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
        temp_err_squared_expr += t ** 2 * s_arg**2
        d = sympy.diff(target_function, arg, evaluate=False)
        temp_exprs.append(t)
        diff_exprs.append(d)
        diff_res_exprs.append(d.subs(target_function, expr).doit())

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
    """
    Calculates the error of the given expression applied to the given values.
    
    Parameters:
        expr (Expr): The sympy expression for which to calculate the error.
        values (Dict[Symbol, Union[Err, ArrayLike]]): A dictionary mapping symbols to Err instances or array-like values.
        
    Returns:
        Err: The computed error as an Err instance.
    """
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