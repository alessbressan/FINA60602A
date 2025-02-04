import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from adjustText import adjust_text
from pathlib import Path
import warnings

import utils_b as optimization
import plotting

warnings.filterwarnings("ignore", category=FutureWarning)

# import data
data_path = str(Path().absolute()) + "/data_will/48_industry_Portfolios.CSV"
df_global = pd.read_csv(data_path, index_col=0)
df_global = df_global / 100
df_global.index = pd.to_datetime(df_global.index, format="%Y%m")  # clean the index to be datetime

# constants
start_date = "2019-12-01"  # 5 years worth of data (60 entries)
risk_free = 0.004  # taken from Kenneth French Library
df_global = df_global.loc[df_global.index >= start_date, :]  # select last 5 years
df = df_global.loc[:, ["Whlsl", "Fin  ", "Util ", "Cnstr", "Hlth "]]  # select 5 assets to work with

N = 6
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

cov = df.cov()
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
target_returns = np.linspace(-8, 8, 202)
x_rf = []
y_rf = []
for ret in target_returns:
    obj_val, weights = optimization.minimize_variance_gurobi(mean, cov, ret, risk_free_asset= True, long_only= True)
    print(obj_val)
    if obj_val is not None:
        x_rf.append(np.sqrt(obj_val))
        y_rf.append(mean @ weights)

# plot
plotting.mean_var_locus(x_rf, y_rf, std, mean, 'Mean Variance Locus (With Risk Free Asset & Short Sale Constraint)')