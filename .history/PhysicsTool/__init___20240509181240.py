from .fmt import round_sigfig, round_err, fmt_err_to_tex
from .logging import log
from .plotting import start_plt, end_plt, err_band_plot
from .solvers import solve_eq, derive_err, calc, calc_err
from .aggregators import err_from_data, avg_err, weighted_avg_err
from . import constants