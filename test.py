import pandas as pd
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from adjustText import adjust_text
from pathlib import Path
import warnings

import optimization
import plotting

warnings.filterwarnings("ignore", category=FutureWarning)

# import data
data_path = str(Path().absolute()) + "/clean_data/48_Industry_Portfolios.CSV"
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index, format="%Y%m")  # clean the index to be datetime

# constants
start_date = "2020-01-01"
N = 48
df = df.loc[df.index >= start_date, :]  # select last 5 years
# df = df / 100  # convert to percentages

# QUESTION 4
N = 48
cov = df.cov()
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
init_w = np.repeat(1/N, N)
target_returns = np.arange(min(mean), max(mean), 0.01)
x = []
y = []
for ret in target_returns:
    response = optimization.minimize_variance(mean, cov, ret, N, False, True, False)
    x.append(response.fun)
    y.append(ret)

# plot
plotting.mean_var_locus(x, y, std, mean, 'Mean Variance Locus (No Risk Free Asset & Short Sale Constraint)')
