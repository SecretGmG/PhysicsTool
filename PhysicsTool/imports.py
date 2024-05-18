import numpy as np
import sympy
import matplotlib.pyplot as plt
import pandas as pd
import scipy

from sympy import symbols
from pandas import read_excel
from sys import set_coroutine_origin_tracking_depth
from copy import deepcopy, copy
from sys import stdout

from .err import Err, calc_err, derive_err
from .logging import log
from .plotting import start_plt, end_plt, err_band_plot
from . import constants

#This line changes the default settings to a bigger font
font = {'size': 12}
plt.rc('font', **font)