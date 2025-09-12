import sympy
import numpy as np
from typing import Iterable


def error_propagation_formula(
    f: sympy.Matrix | Iterable[sympy.Expr] | sympy.Expr, args: list[sympy.Symbol]
) -> tuple[sympy.Expr, sympy.MatrixSymbol]:
    """
    Computes symbolic error propagation A K A^T and returns it with symbolic covariance K.
    should not be used direkctly most of the time, use propagate_error instead.

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
