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

# ____________________ QUESTION B2 ____________________________

# import data
data_path = str(Path().absolute()) + "/data_will/48_industry_Portfolios.CSV"
df_global = pd.read_csv(data_path, index_col=0)
df_global.index = pd.to_datetime(df_global.index, format="%Y%m")  # clean the index to be datetime

# constants
start_date = "2019-12-01"  # 5 years worth of data (60 entries)
N = 48
risk_free = 0.4  # taken from Kenneth French Library
df_global = df_global.loc[df_global.index >= start_date, :]  # select last 5 years
df = df_global.loc[:, ["Whlsl", "Fin  ", "Util ", "Cnstr", "Hlth "]]  # select 5 assets to work with

# QUESTION 1
# first and second moment
cov = df.cov()
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
N = 5
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-5, 5, 0.1)

response_no_rf, weights = optimization.mean_var_portfolio(df, target_returns, N, False, False, False, "gurobi", cardinality= True)
plotting.mean_var_locus(response_no_rf[0], response_no_rf[1], std, mean, 'Mean Variance Locus (No Risk Free Asset)')

# QUESTION 2
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

response_rf, weights = optimization.mean_var_portfolio(df_rf, target_returns, N, True, False, False, "gurobi", cardinality= True)
plotting.mean_var_locus(response_rf[0], response_rf[1], std, mean, 'Mean Variance Locus (With Risk Free Asset)')

# QUESTION 3
N = 6
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-25, 25, 0.4)

# the target return becomes a meaningless parameter when dealing with the tangency portfolio
response = optimization.tangency_portfolio(df, 0.4, False)
# response = optimization.minimize_variance(mean, cov, 0.05, N, True, False, True)

print(response.x)
print(response.x @ mean[1:])
print(np.sqrt(response.x @ cov @ response.x))

w, w_rfr = optimization.analytical_mean_var(mean[1:], cov, 0.05, 0.4, 5, True, True)

# display the weights
fig, ax = plt.subplots()
ax.grid()
x = np.arange(len(mean.index[1:]))
width = 0.4

ax.bar(x= x - width/2, height= w, width= width, label="Analytical Tangency Portfolio Weights")
ax.bar(x= x + width/2, height= response.x, width= width, alpha=0.5, color="red",
       label="Numerically Maximized Sharpe Ratio Weights")
ax.set_title("Weight of Each Asset in the Tangency Portfolio")
ax.set_xticklabels(mean.index[1:])
ax.set_xlabel("Assets")
ax.set_ylabel("Weight")
plt.xticks(rotation=90)
plt.legend()
plt.show()

# display the tangency portfolio on the mean variance locus
tangency_mean = w @ mean[1:]
tangency_std = np.sqrt(w @ cov @ w)

print(w)
print(tangency_std)
print(tangency_mean)
plotting.tangency_plot(response_rf[0], response_rf[1], response_no_rf[0], response_no_rf[1], tangency_mean,
                       tangency_std, std, mean, "Tangency Portfolio")

# QUESTION 4
N = 5
cov = df.cov()
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-8, 8, 0.05)
x_no_rf = []
y_no_rf = []
for ret in target_returns:
    response = optimization.minimize_variance_gurobi_card(mean, cov, ret, False, True, False)
    x_no_rf.append(response.ObjVal)
    y_no_rf.append(mean @ response.X)

# plot
plotting.mean_var_locus(x_no_rf, y_no_rf, std, mean, 'Mean Variance Locus (No Risk Free Asset & Short Sale Constraint)')

# QUESTION 5
N = 6
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-8, 8, 0.02)
x_rf = []
y_rf = []
for ret in target_returns:
    response = optimization.minimize_variance(mean, cov, ret, N, True, True, False)
    x_rf.append(response.fun)
    y_rf.append(mean @ response.x)

# plot
plotting.mean_var_locus(x_rf, y_rf, std, mean, 'Mean Variance Locus (With Risk Free Asset & Short Sale Constraint)')

# QUESTION 6
N = 6
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)

# the target return becomes a meaningless parameter when dealing with the tangency portfolio
response = optimization.tangency_portfolio(df, 0.4, True)

w = np.array(response.x) / sum(response.x)

print(w)
print(w @ mean[1:])
print(np.sqrt(w @ cov @ w))

fig, ax = plt.subplots()
ax.bar(x=mean[1:].index, height=w)
ax.set_title("Weights of Each Asset in Long Only Tangency Portfolio")
ax.set_xlabel("Assets")
ax.set_ylabel("Weight")
ax.grid()
plt.xticks(rotation=90)
plt.show()

plotting.tangency_plot(x_rf, y_rf, x_no_rf, y_no_rf, w @ mean[1:], np.sqrt(w @ cov @ w), std, mean,
                       "Long Only Tangency Portfolio")