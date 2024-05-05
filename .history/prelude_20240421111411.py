import numpy as np
import sympy
import matplotlib.pyplot as plt
import pandas as pd
import .constants

from sympy import symbols
from pandas import read_excel
from sys import set_coroutine_origin_tracking_depth
from copy import deepcopy, copy

from fmt import fmt_err_to_tex
from logging import log
from plotting import start_plt, end_plt, err_band_plot
from solvers import solve_eq, calc, calc_err, err_from_data, avg_err, derive_err

#This line changes the default settings to a bigger font
font = {'size': 12}
plt.rc('font', **font)