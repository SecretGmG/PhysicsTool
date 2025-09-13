import sympy
import numpy as np
from typing import Iterable, Optional, Callable


def error_propagation_formula(
    f: sympy.Matrix | Iterable[sympy.Expr] | sympy.Expr, args: list[sympy.Symbol]
) -> tuple[sympy.Expr, sympy.MatrixSymbol]:
    """
    Computes symbolic error propagation A K A^T and returns it with symbolic covariance K.
    should not be used directly most of the time, use propagate_error instead.

    Args:
        f: Vector-valued expression (Matrix, iterable, or expression).
        args: Variables for Jacobian computation.

    Returns:
        (Expression for propagated error, symbolic covariance matrix K).
    """

    if not isinstance(f, sympy.Matrix):
        if isinstance(f, sympy.Expr):
            f = [f]

        f = sympy.Matrix(list(f))

    A = f.jacobian(args)
    K = sympy.MatrixSymbol("K", len(args), len(args))
    # these are sympy expressions, so we need the asteriks * to multiply them
    # this is the same as A @ K @ A.T in numpy, but sympy doesn't support the @ operator
    return A * K * A.T, K  # skript S.12


def propagate_error(
    f: sympy.Matrix | Iterable[sympy.Expr] | sympy.Expr,
    args_symbols: Iterable[sympy.Symbol],
    args: np.ndarray,
    cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagates error through a function using the covariance matrix.

    Args:
        f: Vector-valued expression (Matrix, iterable, or expression).
        args_symbols: Variables for Jacobian computation.
        args: Values of the variables along the first dimension.
        cov: Covariance matrix of the variables.

    Returns:
        (Result of the function, propagated covariance matrix).
    """
    cov_expr, K = error_propagation_formula(f, list(args_symbols))
    result = sympy.lambdify(args_symbols, f)(*args)
    result_cov = sympy.lambdify([*args_symbols, K], cov_expr)(*args, cov)
    return (result, result_cov)


def derive_err(
    expr: sympy.Expr,
    values: Optional[list[sympy.Symbol]] = None,
    target_symbol: sympy.Symbol = sympy.Symbol("f"),
    err_prefix: str = "s",
    relative: bool = False,
    logger: Callable = lambda *args: None,
) -> sympy.Expr:
    """
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
        logger (Optional[logger]): Callable that logs equations describing the errror derrivation.

    Returns:
        Expr: The sympy expression for the Gaussian error in `expr`.

    Example:
        To calculate the error of an expression with relative errors:

            x, y = sympy.symbols('x y')
            expr = x * y
            error_expr = derive_err(expr, relative=True)
    """

    target_err_symbol = sympy.Symbol(f"{err_prefix}_{target_symbol}")

    target_err_symbol_squared = target_err_symbol**2

    if values is not None:
        free_args = list(set(expr.free_symbols) & set(values))
    else:
        free_args = list(expr.free_symbols)

    target_function = sympy.Function(target_symbol)(*free_args)
    temp_err_squared_expr = sympy.S.Zero
    temp_exprs, diff_exprs, diff_res_exprs = [], [], []

    for arg in free_args:
        t, s_arg = sympy.symbols(f"temp_{arg}, {err_prefix}_{arg}")
        if relative:
            if not isinstance(arg, sympy.Symbol):
                raise Exception("Encountered non Symbol in free_symbols")
            temp_err_squared_expr += t**2 * s_arg**2 * arg**2
        else:
            temp_err_squared_expr += t**2 * s_arg**2
        d = sympy.diff(target_function, arg, evaluate=False)
        temp_exprs.append(t)
        diff_exprs.append(d)
        diff_res_exprs.append(d.subs(target_function, expr).doit())

    if relative:
        with sympy.evaluate(False):
            temp_err_squared_expr /= expr**2

    diff_err_squared_expr = temp_err_squared_expr.subs(
        list(zip(temp_exprs, diff_exprs)), evaluate=False
    )

    logger(sympy.Eq(target_err_symbol_squared, diff_err_squared_expr, evaluate=False))

    with sympy.evaluate(False):
        err_squared_expr = temp_err_squared_expr.subs(zip(temp_exprs, diff_res_exprs))

        logger(sympy.Eq(target_err_symbol_squared, err_squared_expr))

    err_expr = sympy.sqrt(err_squared_expr)

    logger(sympy.Eq(target_err_symbol, err_expr, evaluate=False))

    return err_expr
