import pandas as pd
import sympy
from typing import Optional, TextIO, Union
from IPython.display import display, Latex
from sys import stdout


def _get_tex_df(df: pd.DataFrame, caption: str = 'Caption', label: str = 'tab:labelname') -> str:
    '''
    Generate LaTeX representation of a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to convert to LaTeX.
        caption (str): The caption for the LaTeX table. Defaults to 'Caption'.
        label (str): The label for the LaTeX table. Defaults to 'tab:labelname'.
    
    Returns:
        str: The LaTeX string representing the DataFrame.
    '''
    if df.empty:
        raise ValueError('The DataFrame is empty and cannot be converted to LaTeX.')
    
    df = df.copy()
    from .err import Err
    df = df.map(
        lambda e: e.latex() if type(e) is Err else e
    )
    h = r'\begin{table}[H]' + '\n' + r'\centering' + '\n'
    t = f'\\caption{{{caption}}}\n\\label{{{label}}}\n' + r'\end{table}' + '\n'
    return h + df.to_latex(index=False, escape=False) + t


def _get_latex_str(expr: Union[str, sympy.Basic, pd.DataFrame, Latex]) -> str:
    '''
    Generate LaTeX representation of an expression.

    Parameters:
        expr (Union[str, sympy.Basic, pd.DataFrame, Latex]): The expression to convert to LaTeX.
    
    Returns:
        str: The LaTeX string representing the expression.
    '''
    latex_str = None
    if isinstance(expr, sympy.Basic):
        latex_str = f'\\[\n{sympy.latex(expr)}\n\\]'
    elif isinstance(expr, pd.DataFrame):
        latex_str = _get_tex_df(expr)
    elif isinstance(expr, Latex):
        latex_str = expr.data
    elif isinstance(expr, str):
        latex_str = expr
    else:
        try:
            latex_str = expr._repr_latex_()
            if isinstance(latex_str, tuple):
                latex_str, *_ = latex_str
        except Exception:  # Catch for missing _repr_latex_
            latex_str = str(expr)
    return latex_str


def log(expr: Union[str, sympy.Basic, pd.DataFrame, Latex], do_display: bool = True, tex: Optional[TextIO] = stdout) -> None:
    '''
    Log the expression and optionally write its LaTeX representation to a TextIO.

    Parameters:
        expr (Union[str, sympy.Basic, pd.DataFrame, Latex]): The expression to log.
        do_display (bool, optional): Whether to display the expression. Defaults to True.
        tex (Optional[TextIO]): A TextIO object to write LaTeX representation. Defaults to sys.stdout.
    '''
    if do_display:
        display(expr)

    if tex:
        tex.write(_get_latex_str(expr) + '\n')