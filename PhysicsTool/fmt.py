import numpy as np
from typing import Tuple, Optional, TextIO
from IPython.display import Latex
from .logging import log

def _get_order(num: float):
    if num == 0:
        return -1024
    return int(np.floor(np.log10(abs(num)))) + 1


def _fmt_to_order(num: float, order: int, min_precision: int, max_precision: int):
    precision = np.clip(-order, min_precision, max_precision)
    return f"{round(num,-order):.{precision}f}"


def round_sigfig(num: float, sigfigs: int = 2):
    """
    Round the number to a specified number of significant figures.

    Parameters:
        num (float): The number to round.
        sigfigs (int, optional): Number of significant figures. Defaults to 2.
    """
    order = _get_order(num) - sigfigs
    return round(num, -order)

round_sigfig = np.vectorize(round_sigfig, otypes=[float])

def round_err(mean: float, err: float, sigfigs: int = 2) -> Tuple[float, float]:
    """
    Round the mean and error to a specified number of significant figures.

    Parameters:
        mean (float): The mean value.
        err (float): The error value.
        sigfigs (int, optional): Number of significant figures. Defaults to 2.

    Returns:
        Tuple[float, float]: The rounded mean and error.
    """
    order = _get_order(err) - sigfigs
    err = round(err, -order)
    mean = round(mean, -order)
    return mean, err

round_err = np.vectorize(round_err, otypes=[float, float])

def fmt_err_to_tex(
    mean: float,
    err: float,
    sigfigs: int = 2,
    min_precision: int = 0,
    max_precision: int = 16,
    relative: bool = False,
    do_display: bool = False,
    tex: Optional[TextIO] = None
):
    """
    Format the mean and error in LaTeX notation with a specified number of significant figures.

    Parameters:
        mean (float): The mean value.
        err (float): The error value.
        sigfigs (int, optional): Number of significant figures. Defaults to 2.
        min_precision (int, optional): Minimum precision for formatting. Defaults to 0.
        max_precision (int, optional): Maximum precision for formatting. Defaults to 16.
        relative: bool = False: If set to true the error will be given in percent. Defaults to False
        do_display (bool, optional): Whether to display the formatted tex expression. Defaults to False.
        tex (Optional[TextIO], optional): A file object to write LaTeX representation into. Defaults to None.

    Returns:
        str: Formatted mean and error in LaTeX notation.
    """

    if err == 0:
        tex_str = _fmt_to_order(mean, _get_order(
            mean) - sigfigs, min_precision, max_precision)

    else:
        mean_str = _fmt_to_order(mean, _get_order(
            err) - sigfigs, min_precision, max_precision)
        if relative:
            err = 100*err/mean
        err_str = _fmt_to_order(err, _get_order(
            err) - sigfigs, min_precision, max_precision)
        if relative:
            err_str += "\\%"
        tex_str = f"${mean_str}\\pm{err_str}$"

    log(Latex(tex_str), do_display, tex)
    return tex_str

fmt_err_to_tex = np.vectorize(fmt_err_to_tex, otypes=[str])