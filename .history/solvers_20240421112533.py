import numpy as np
import sympy
from sympy import Symbol, Eq, Function, Expr
from .logging import log
from .fmt import fmt_err_to_tex


def solve_eq(eq: Eq, target_symbol: Symbol, do_display: bool = False, tex = None) -> Expr:
    """
    Solve the given equation for the target symbol.

    Parameters:
        eq (Eq): The equation to solve.
        target_symbol (Symbol): The symbol to solve for.
        do_display (bool, optional): Whether to display intermediate steps. Defaults to False.
        tex (Optional[TextIO], optional): A file object to write LaTeX representation into. Defaults to None.

    Returns:
        Expr: The solution expression for the target symbol.
    """
    log(f'solve the following expression for {target_symbol}')
    log(eq, do_display, tex)
    expr = sympy.solve(eq, target_symbol)[0]
    log(f'solving for {target_symbol}, we get the expression', do_display, tex)
    log(Eq(target_symbol, expr), do_display, tex)
    return expr


def derive_err(
    expr: Expr,
    values = None,
    target_symbol: Symbol = Symbol('f'),
    err_prefix: str = 's',
    do_display: bool = False,
    tex = None
) -> Expr:
    """
    Calculate the absolute Gaussian error.

    This function calculates the absolute Gaussian variance for the given expression.

    Parameters:
        expr Expr: The expression to calculate the absolute Gaussian variance for. 
        measured_values (List[Symbol]): The list of symbols representing measured values from an experiment.
        target_symbol (Optional[Symbol]): The symbol of the expression to calculate the Gaussian variance for.
        err_prefix (str): The string every error term is prefixed with
        verbose (bool): Whether to display intermediate steps. Defaults to False.
        tex (Optional[TextIO]): A file object to write LaTeX representation into. Defaults to None.

    Returns:
        Expr: The absolute Gaussian error
    """
    target_err_symbol = Symbol(f'{err_prefix}_{target_symbol}')
    target_err_symbol_squared = target_err_symbol ** 2

    if values:
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

    # log(f'Taking the derivatives we get', do_display, tex)
    # for lhs, rhs in zip(diff_exprs, diff_res_exprs):
    #    log(Eq(lhs, rhs, evaluate=False), do_display, tex)

    with sympy.evaluate(False):
        # log('therefore', do_display, tex)
        err_squared_expr = temp_err_squared_expr.subs(
            zip(temp_exprs, diff_res_exprs))
        log(Eq(target_err_symbol_squared, err_squared_expr), do_display, tex)

    err_expr = sympy.sqrt(err_squared_expr)
    log(Eq(target_err_symbol, err_expr, evaluate=False), do_display, tex)
    return err_expr


def err_from_data(array: np.ndarray, axis=None):
    """
    Calculate the mean and standard error of the mean from data.

    Parameters:
        array (np.ndarray): The input data array.
        axis (Optional[int]): The axis along which the mean and error are calculated.
            If None, calculates the mean and error over the entire array. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The mean and standard error of the mean.
    """
    return np.mean(array, axis=axis), np.std(array, axis=axis) / np.sqrt(np.size(array, axis=axis))


def avg_err(mean: np.ndarray, err: np.ndarray, axis=None):
    """
    Calculate the weighted average and combined error from means and errors.

    Parameters:
        mean (np.ndarray): The array of mean values.
        err (np.ndarray): The array of errors corresponding to the mean values.
        axis (Optional[int]): The axis along which the average and error are calculated.
            If None, calculates the average and error over the entire array. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The weighted average and combined error.
    """
    return np.mean(mean, axis=axis), np.sqrt(np.sum(err**2, axis=axis)) / np.size(err, axis=axis)


def calc(
        expr: Expr,
        values,
        target_symbol: Symbol = Symbol('x'),
        do_display: bool = False,
        tex = None
):
    log(Eq(target_symbol, expr), do_display, tex)

    for symbol, value in values.items():
        values[symbol] = np.array(value, dtype=np.double)
    
    if set(expr.free_symbols) <= set(values.keys()):
        res_func = sympy.lambdify(values.keys(), expr)
        res = res_func(*(values.values()))

        if res is float:
            fmt_err_to_tex(res, 0, do_display=do_display, tex=tex)
    else:
        print("WARNING: not all free symbols were defined therefore an expression was returned!")
        values = {symbol: value for symbol,
                       value in values.items() if value.size == 1}
        res = expr.subs(values).doit()
    
    log(Eq(target_symbol, res), do_display, tex)
    return res

def calc_err(
    expr: Expr,
    values,
    target_symbol: Symbol = Symbol('f'),
    err_prefix: str = 's',
    do_display: bool = False,
    tex = None
):
    """
    Calculate mean and error for a given expression and values.

    Parameters:
        expr (Expr): The expression for which to calculate the mean.
        values (Dict[Symbol, Union[float, Tuple[float, float]]]): Dictionary mapping symbols to their values or mean/error tuples.
        target_symbol (Symbol, optional): The symbol of the mean value. Defaults to Symbol('f').
        err_prefix (str, optional): Prefix for the error symbol. Defaults to 's'.
        do_display (bool, optional): Whether to display intermediate steps. Defaults to False.
        tex (Optional[TextIO], optional): A file object to write LaTeX representation into. Defaults to None.

    Returns:
        Tuple[float, float]|Tuple[Expr, Expr]: The mean and error, or expressions if there are undefined symbols.
    """
    mean_values = {}
    err_values = {}
    const_values = {}

    for symbol, value in values.items():
        if hasattr(value, '__len__') and len(value) == 2:
            mean, err = value[0], value[1]
            mean_values[symbol] = np.array(mean, dtype=np.double)
            err_values[Symbol(f"{err_prefix}_{symbol}")
                       ] = np.array(err, dtype=np.double)
        else:
            const = value
            const_values[symbol] = np.array(const, dtype=np.double)

    err_expr = derive_err(expr, mean_values.keys(),
                          target_symbol, err_prefix, do_display, tex)

    mean_values = mean_values | const_values
    err_values = mean_values | const_values | err_values

    target_err_symbol = Symbol(f'{err_prefix}_{target_symbol}')

    mean, err = None, None

    if set(err_expr.free_symbols) <= set(err_values.keys()) and set(expr.free_symbols) <= set(mean_values.keys()):
        mean_func = sympy.lambdify(mean_values.keys(), expr)
        err_func = sympy.lambdify(err_values.keys(), err_expr)

        mean = mean_func(*(mean_values.values()))
        err = err_func(*(err_values.values()))
        if mean is float and err is float:
            fmt_err_to_tex(mean, err, do_display=do_display, tex=tex)
    else:
        print("WARNING: not all free symbols were defined therefore an expression was returned!")
        mean_values = {symbol: value for symbol,
                       value in mean_values.items() if value.size == 1}
        err_values = {symbol: value for symbol,
                      value in err_values.items() if value.size == 1}
        mean = expr.subs(mean_values).doit()
        err = err_expr.subs(err_values).doit()
        log(Eq(target_symbol, mean, evaluate=False), do_display, tex)
        log(Eq(target_err_symbol, err, evaluate=False), do_display, tex)
    return np.array((mean, err))