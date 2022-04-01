"""Environment module

Commonly used packages and default settings are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


### Internal packages
import sys
import os
from os.path import join, isdir, isfile, exists, basename, dirname, split, abspath
import shutil
from glob import glob
import datetime
import joblib
import json
import re
from itertools import product
from time import time, sleep
from collections import defaultdict
from copy import deepcopy as copy
from tqdm import tqdm, trange
import warnings
from contextlib import ContextDecorator
from dataclasses import dataclass
from IPython.display import display, Markdown
import subprocess


### External packages
import numpy as np
import pandas as pd
from tabulate import tabulate
from numba import njit, cuda
from dask import delayed, compute
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from switch import Switch
from parse import parse, search
import missingno as msno


## Plot packages
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.gridspec import GridSpec
import seaborn as sns


## Matplotlib options
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.rc('font', family='DejaVu Sans')
plt.rc('axes', unicode_minus=False)
plt.rc('font', size=20)
plt.rc('figure', titlesize=40, titleweight='bold', figsize=(16, 8))
plt.style.use('ggplot')


### Set options
np.set_printoptions(suppress=True, precision=6, edgeitems=20, linewidth=100, formatter={"float": lambda x: "{:.3f}".format(x)})
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 100)


### Warning
warnings.filterwarnings('ignore')
