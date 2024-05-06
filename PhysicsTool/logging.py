import pandas as pd
import sympy
from typing import Optional, TextIO
from IPython.display import display, Latex
from sys import stdout


def _get_tex_df(df: pd.DataFrame, do_display: bool, tex: Optional[TextIO]):
    h = r"\begin{table}[H]" + "\n" + r"\centering" + "\n"
    t = r"\caption{Caption}" + "\n" r"\label{tab:labelname}" + \
        "\n" + r"\end{table}" + '\n'
    return h+df.to_latex(index=False, escape=False)+t


def _get_tex_sympy(expr: sympy.core.Basic, do_display: bool, tex: Optional[TextIO]):
    return f"\\[\n{sympy.latex(expr)}\n\\]"


def log(expr: str | sympy.Basic | pd.DataFrame | Latex, do_display: bool = True, tex: Optional[TextIO] = stdout) -> None:
    """
    Log the expression and optionally write its LaTeX representation to a TextIO.
    Parameters:
        expr str: The expression to log.
        verbose (bool, optional): Whether to display the expression. Defaults to True.
        tex (Optional[TextIO]): A TextIO object to write LaTeX representation. Defaults to sys.stdout.
    """
    if do_display:
        display(expr)

    latex_str = None

    if (isinstance(expr, sympy.Basic)):
        latex_str = _get_tex_sympy(expr, do_display, tex)
    elif (isinstance(expr, pd.DataFrame)):
        latex_str = _get_tex_df(expr, do_display, tex)
    elif (isinstance(expr, Latex)):
        latex_str = expr._repr_latex_()
        if type(latex_str) is tuple:
            latex_str, *_ = latex_str
    else:
        latex_str = str(expr)

    if tex:
        tex.write(latex_str + '\n')