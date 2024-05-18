import numpy as np
import sympy
from .logging import log
from numpy.typing import ArrayLike
from sympy import Symbol, Expr, Function, Eq
from typing import Optional, Dict, List, TextIO, Self

class Err:
    def __init__(self, mean: ArrayLike, err : ArrayLike):
        mean, err = np.broadcast_arrays(mean, err)
        self.mean = np.array(mean)
        self.err = np.array(err)

    def from_data(data: ArrayLike, axis=None) -> Self:
        return Err(np.mean(data, axis), np.std(data, axis, ddof=1) /
                   np.sqrt(np.size(data, axis)))

    def apply(self, foo: Function | Expr) -> Self:
        """
        Applies the given function or expression to Self. 
        If given an expression it can only have a single free symbol.
        """
        x = Symbol('x')
        if isinstance(foo, sympy.FunctionClass):
            foo = foo(x)
        else:
            free_symbols = foo.free_symbols
            assert len(free_symbols) == 1, \
                'cannot apply function with more than one argument. Use calc_err to compute such expressions'
            x = list(free_symbols)[0]
        return Err(sympy.lambdify(x, foo)(self.mean), sympy.lambdify(x, sympy.Abs(sympy.diff(foo, x)))(self.mean)*self.err)

    def approxEq(self, other) -> bool:
        """
        returns True if both values lie within each others error bounds
        """
        if isinstance(other, Err):
            return np.allclose(self.mean, other.mean, atol=np.minimum(self.err, other.err))
        else:
            return np.allclose(self.mean, other, atol=self.err)

    def average(self, axis=None) -> Self:
        """
        computes the average and its propagated error along a given axis
        """
        return Err(self.mean.mean(axis = axis), np.sqrt(np.sum(self.err**2, axis=axis)) / np.size(self.err, axis=axis))

    def weighted_average(self, axis=None) -> Self:
        """
        Calculate the weighted average and combined error along a given axis as explained here:
        https://www.physics.umd.edu/courses/Phys261/F06/ErrorPropagation.pdf
        """
        return Err(np.sum(self.mean*self.err**-2, axis=axis)/np.sum(self.err**-2), np.sum(self.err**-2, axis=axis)**-0.5)

    def latex(self,
              sigfigs: int = 2,
              min_precision: int = 0,
              max_precision: int = 16,
              relative: bool = False
              ) -> ArrayLike:
        """
        Returns a numpy string array with a latex expression for every element
        """
        mean_str, err_str = self.formatted(
            sigfigs, min_precision, max_precision, relative)
        if relative:
            err_str += '\\%'
        return '$' + mean_str + '\pm' + err_str + '$'

    def _repr_latex_(self):
        return '$'+r'\\'.join(l[1:-1] for l in np.ravel(np.array([self.latex()])))+'$'
        mean_str, err_str = self.formatted()
        return '$' + r'\\'.join(rf'{m} \pm {e}' for m, e in zip(np.ravel(mean_str), np.ravel(err_str))) + '$'

    def formatted(self,
                   sigfigs: int = 2,
                   min_precision: int = 0,
                   max_precision: int = 16,
                   relative: bool = False,
                   ) -> ArrayLike:

        temp_err = self.err.copy()
        if relative:
            temp_err *= 100/self.mean

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            zeroformatted = _fmt_to_order(self.mean, _get_order(
                self.mean) - sigfigs, min_precision, max_precision)
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
        return self.formatted(sigfigs, min_precision, max_precision, relative)[0]
    
    def formatted_err(self,
                   sigfigs: int = 2,
                   min_precision: int = 0,
                   max_precision: int = 16,
                   relative: bool = False,
                   ) -> ArrayLike:
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

    def __repr__(self):
        mean_str, err_str = self.formatted()
        return f'{mean_str} ± {err_str}'

@np.vectorize(otypes = [int])
def _get_order(num: float) -> int:
    if num == 0:
        return -1024
    return int(np.floor(np.log10(abs(num)))) + 1


@np.vectorize(otypes = ['O'])
def _fmt_to_order(num: float, order: int, min_precision: int, max_precision: int) -> str:
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
    Returns:
        Err: the computed error
    """
    err_prefix = 'temporary_error_prefix'

    err_mean_values = {}
    err_err_values = {}
    const_values = {}
    for key, val in values.items():
        if isinstance(val, Err):
            err_mean_values[key] = values[key].mean
            err_err_values[Symbol(f'{err_prefix}_{key}')] = values[key].err
        else:
            const_values[key] = np.array(values[key])

    values = err_mean_values | err_err_values | const_values

    err_expr = derive_err(expr, err_mean_values.keys(), err_prefix=err_prefix)

    mean_func = sympy.lambdify(values.keys(), expr)
    err_func = sympy.lambdify(values.keys(), err_expr)

    return Err(mean_func(*values.values()), err_func(*values.values()))