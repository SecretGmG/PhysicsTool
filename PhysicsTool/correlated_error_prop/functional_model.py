from abc import ABC, abstractmethod
from copy import deepcopy
import warnings
from collections.abc import Callable
import matplotlib.pyplot as plt
import scipy.sparse
import sympy
from scipy import stats
import numpy as np


class FunctionalModel(ABC):
    """
    Abstract base class for parametric models fitted with weighted least squares.

    Stores all intermediate computations for transparency and debugging.
    """

    ### Constants
    NR_TICKS_OVERCROWDING_TRESHOLD = 10
    ### Public fields

    # defines break conditions for the iterative process
    max_iter: int = 10_000
    epsilon: float = 1e-4

    # used to provide nicer outputs, needs to be set by the implementation
    parameter_symbols: list[sympy.Symbol]

    # is called after each iteration, can be used to print or log the current state of the model
    # default is a no-op, but can be set to a function that takes the model as argument
    logger: Callable = lambda *args: None

    # defines the initial parameters, can be set before calling fit() to define initial parameters
    # default value is determined by implementation
    parameters: np.ndarray

    ### Readonly fields
    iterations: int = 0

    x: np.ndarray  # observed x
    y: np.ndarray  # observed y
    P: scipy.sparse.spmatrix  # weighting matrix

    A: np.ndarray  # design matrix

    normal_matrix: np.ndarray
    b: np.ndarray  # rhs of normal equation
    delta_parameters: np.ndarray  # updates of the parameters in iterative least squares

    m_0: float = np.inf

    y_pred: np.ndarray

    residuals: np.ndarray

    @property
    def dof(self):
        return len(self.x) - len(self.parameters)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight_matrix: scipy.sparse.spmatrix | None = None,
    ):
        """
        Fits model to data using iterative weighted least squares.
        Args:
            x (np.ndarray) : the independent variables
            y (np.ndarray) : the dependent variables
            weight_matrix (scipy.sparse.spmatrix) : the weights associated to the dependent variables, should be equal to the inverse of the Cofactormatrix
                if None defaults to the identity matrix. Default is None.
        """

        if self.parameters is None:
            raise ValueError(
                "Cannot fit, because self.parameters is None, set your initial parameters!"
            )

        self.x = np.asarray(x)
        self.y = np.asarray(y)

        # by default use the identity matrix as weight matrix
        if weight_matrix is None:
            weight_matrix = scipy.sparse.diags(np.ones(len(y)))

        weight_matrix = scipy.sparse.csr_matrix(weight_matrix)

        self.P = weight_matrix

        # to start the iteration process the model needs to be evaluated at the initial parameters
        self.y_pred = self.eval(x)

        for i in range(self.max_iter):
            self.iterations = i + 1

            self.A = self.get_design_matrix(x)

            self.normal_matrix = self.A.T @ self.P @ self.A
            # use P.dot() since it is more efficient here to compute from right to left
            # because the intermediate matrix doesn't need to be stored and the sparse matrix might improve performance
            self.b = self.A.T @ self.P.dot(self.y - self.y_pred)

            # use lstsq instead of inv to avoid computing the inverse and handle singular normal matrices
            self.delta_parameters = np.linalg.lstsq(self.normal_matrix, self.b)[0]

            # Update the parameters and associated values
            self.parameters = self.parameters + self.delta_parameters
            self.y_pred = self.eval(self.x)
            self.residuals = self.y - self.y_pred
            self.m_0 = np.sqrt(
                (self.residuals.T @ self.P @ self.residuals) / (self.dof)
            )

            # finally call the logger
            self.logger(self)

            if np.all(np.abs(self.delta_parameters) < self.epsilon):
                break

        if not np.all(np.abs(self.delta_parameters) < self.epsilon):
            warnings.warn(
                "The Functional Model did not converge, make sure you set reasonable initial parameters"
            )

    def parameter_cof(self) -> np.ndarray:
        """
        Returns the cofactor matrix of the parameters.
        NOTE: needs to be multiplied by m_0**2 or sigma_0**2 to get the covariance matrix
        """
        return np.linalg.inv(self.normal_matrix)

    def parameter_corr(self) -> np.ndarray:
        """
        Returns the correlation matrix of the parameters.
        """
        cofactor_matrix = self.parameter_cof()
        diag = cofactor_matrix.diagonal() ** 0.5
        return cofactor_matrix / np.outer(diag, diag)

    def eval_cof(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the cofactor matrix of the predicted values at the given input x.
        NOTE: needs to be multiplied by m_0**2 or sigma_0**2 to get the covariance matrix
        """
        A = self.get_design_matrix(x)
        return A @ self.parameter_cof() @ A.T

    def eval_stderr(self, x: np.ndarray, sigma) -> np.ndarray:
        """
        Returns only the diagonal of the covariance of the prediction at x. sigma determines the scale of the covariance, default is m_0.
        """
        if sigma is None:
            sigma = self.m_0
        A = self.get_design_matrix(x)
        # if we only care about the diagonal, it is more efficient to compute the sum directly
        return sigma * np.einsum("ij,jk,ki -> i", A, self.parameter_cof(), A.T) ** 0.5

    def chi2_threshold(self, alpha: float = 0.05) -> float:
        """
        Returns the critical value for the reduced chi-squared test at significance level alpha.

        This value is (X^2 / dof), where X^2 is the critical value from the chi-squared distribution with dof = degrees of freedom.

        To test the model, compare the ratio m_0^2 / sigma_0^2 (i.e. the reduced chi-squared statistic) to this threshold.
        If it is greater, the model is rejected at significance level alpha.

        NOTE: Smaller alpha means stricter evidence is required to reject the model,
        not a tolerance for larger errors.
        """
        return float(stats.chi2.ppf(1 - alpha, self.dof) / self.dof)

    def plot_prediction(self, sigma=None, errorbar=True, kwargs=None):
        """
        Plots the model prediction with error bars.
        """
        if sigma is None:
            sigma = self.m_0
        if kwargs is None:
            kwargs = {
                "marker": ".",
                "label": "model prediction",
                "color": "black",
                "alpha": 0.5,
            }

        if not errorbar:
            plt.scatter(self.x, self.y_pred, **kwargs)
        else:
            y_stderr = self.eval_stderr(self.x, sigma)
            plt.errorbar(self.x, self.y_pred, y_stderr, linestyle="", **kwargs)

    def plot_prediction_smooth(
        self,
        sigma=None,
        errorband=True,
        n_points=200,
        plt_kwargs=None,
        fill_kwargs=None,
    ):
        """
        Plots the model prediction as a smooth line with error band.
        """
        if sigma is None:
            sigma = self.m_0
        if plt_kwargs is None:
            plt_kwargs = {"color": "black", "label": "model prediction", "alpha": 0.5}
        if fill_kwargs is None:
            fill_kwargs = {"color": "black", "alpha": 0.2, "label": "error band"}

        linspace = np.linspace(self.x.min(), self.x.max(), n_points)
        y = self.eval(linspace)
        plt.plot(linspace, y, **plt_kwargs)

        if errorband:
            eval_stderr = self.eval_stderr(linspace, sigma)
            plt.fill_between(linspace, y - eval_stderr, y + eval_stderr, **fill_kwargs)

    def show_correlation(self, parameter_ticks: bool | None = None):
        """
        Plots the correlation matrix of the parameters.
        """
        matshow = plt.matshow(self.parameter_corr())
        plt.colorbar(matshow, label="correlation")

        # set the ticks to the parameter symbols if they are available
        # and the number of parameters is small enough
        if parameter_ticks is None:
            if self.parameter_symbols is None:
                parameter_ticks = False
            else:
                parameter_ticks = (
                    len(self.parameter_symbols) < self.NR_TICKS_OVERCROWDING_TRESHOLD
                )

        if parameter_ticks:
            n_ticks = len(self.parameter_symbols)
            ticks = [f"${sympy.latex(s)}$" for s in self.parameter_symbols]
            plt.xticks(range(n_ticks), ticks)
            plt.yticks(range(n_ticks), ticks)

        plt.gca().xaxis.set_label_position("top")
        plt.xlabel("parameter")
        plt.ylabel("parameter")

    def print_parameters(self, precision: int = 3, sigma: float | None = None):
        """prints the parameters and their uncertainties in a human-readable format.

        Args:
            precision int: Defaults to 3.
            sigma float: Defaults to None.
        """
        if sigma is None:
            sigma = self.m_0
        errors = sigma * np.sqrt(np.diag(self.parameter_cof()))
        for i, (v, e) in enumerate(zip(self.parameters, errors)):
            if self.parameter_symbols is not None:
                s = self.parameter_symbols[i]
            else:
                s = sympy.Symbol(f"a_{i}")
            print(f"{s} = {v:.{precision}f} ± {e:.{precision}f}")

    def print_parameters_latex(self, precision: int = 3, sigma: float | None = None):
        """
        Prints the parameters and their uncertainties in LaTeX format.
        sigma is the scale of the covariance, default is m_0.
        Args:
            precision: number of decimal places to print, default is 3.
        """
        if sigma is None:
            sigma = self.m_0
        errors = sigma * np.sqrt(np.diag(self.parameter_cof()))
        for i, (v, e) in enumerate(zip(self.parameters, errors)):
            if self.parameter_symbols is not None:
                s = self.parameter_symbols[i]
            else:
                s = sympy.Symbol(f"a_{i}")
            print(f"${sympy.latex(s)} = \\SI{{{v:.{precision}f}({e:.{precision}f})}}$")

    def copy(self):
        """
        Returns a deep copy of the model.
        """
        return deepcopy(self)

    @abstractmethod
    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the model predictions at the given x values. Needs Implementation.
        """
        pass

    @abstractmethod
    def get_design_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Computes and returns the design matrix for the inputs x. Needs Implementation.
        """
        pass


class PolyFunctionalModel(FunctionalModel):
    degree: int

    def __init__(self, degree: int):
        self.degree = degree

        # set initial parameters to zero by default
        self.parameters = np.zeros(degree + 1)

        self.parameter_symbols = [
            sympy.Symbol(f"a_{i}") for i in reversed(range(degree + 1))
        ]

        self.max_iter = 1  # this is a linear model, so only one iteration is needed to fit the parameters
        self.epsilon = np.inf  # no need to check for convergence

    def get_design_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([x**i for i in reversed(range(self.degree + 1))])

    def eval(self, x):
        return np.polyval(self.parameters, x)


class SympyFunctionalModel(FunctionalModel):
    function_expr: sympy.Expr
    feature_symbol: sympy.Symbol

    differential_expressions: list[sympy.Expr]
    differentials: list[Callable]  # store these for debugging and transparancy
    lambdified: Callable

    def __init__(
        self,
        function_expr: sympy.Expr,
        parameter_symbols: list[sympy.Symbol],
        feature_symbol: sympy.Symbol,
    ):
        self.function_expr = function_expr
        self.parameter_symbols = parameter_symbols
        self.feature_symbol = feature_symbol

        self.lambdified = sympy.lambdify(
            [*parameter_symbols, self.feature_symbol], self.function_expr
        )

        # compute the partial derivative of the function with respect to each parameter
        self.differential_expressions = [
            sympy.diff(self.function_expr, a) for a in parameter_symbols
        ]
        self.differentials = [
            sympy.lambdify([*parameter_symbols, self.feature_symbol], diff)
            for diff in self.differential_expressions
        ]

        # set initial parameters to zero by default
        self.parameters = np.zeros(len(parameter_symbols))

    def eval(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.broadcast_to(self.lambdified(*self.parameters, x), x.shape)

    def get_design_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        # the columns of the design matrix are the partial derivatives of the function with respect to each parameter
        # evaluated at the input x, and the current parameters
        # broadcasting the result is necessary because sometimes the lambdified function returns a scalar (if the function is constant)
        A_columns = [
            np.broadcast_to(d(*self.parameters, x), x.shape) for d in self.differentials
        ]

        return np.column_stack(A_columns)
