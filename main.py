import pandas as pd
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from matplotlib import pyplot as plt
from adjustText import adjust_text
from pathlib import Path
from tqdm import tqdm

import optimization
import plotting

# import data
data_path = str(Path().absolute()) + "/data_will/48_industry_Portfolios.CSV"
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index, format="%Y%m")  # clean the index to be datetime

# constants
start_date = "2019-12-01"  # 5 years worth of data (60 entries)
N = 48
risk_free = 0.4  # taken from Kenneth French Library
df = df.loc[df.index >= start_date, :]  # select last 5 years
# df = df / 100  # convert to percentages


# QUESTION 1
# first and second moment
cov = df.cov()
print(np.linalg.det(cov))
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-8, 8, 0.5)

# scipy methodology
"""
x = []
y = []
for ret in target_returns:
    response = optimization.minimize_variance(mean, cov, ret, N, False, False, False)
    x.append(response.fun)
    y.append(ret)

# plot
plotting.mean_var_locus(x, y, std, mean, 'Mean Variance Locus (No Risk Free Asset)')
"""

# analytical methodology
"""
inv_sigma = np.linalg.pinv(df.cov())
mean = df.mean(axis=0)
A = np.ones(48).T @ inv_sigma @ np.ones(48)
B = np.ones(48).T @ inv_sigma @ mean
C = mean.T @ inv_sigma @ mean
delta = A * C - B ** 2

results = []
for ret in target_returns:
    w = (C - ret * B) / delta * inv_sigma @ np.ones(48) + (ret * A - B) / delta * inv_sigma @ mean
    results.append([np.sqrt(w.T @ df.cov() @ w), mean.T @ w])

results = np.array(results)
plotting.mean_var_locus(results[:,0], results[:,1], std, mean, 'Mean Variance Locus (No Risk Free Asset)')
"""


response = optimization.mean_var_portfolio(df, target_returns, N, False, "gurobi")
plotting.mean_var_locus(response[0], response[1], std, mean, 'Mean Variance Locus (No Risk Free Asset)')


# QUESTION 2
N = 49
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))


# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-8, 8, 0.5)
"""
x = []
y = []
for ret in target_returns:
    response = optimization.minimize_variance(mean, cov, ret, N, True, False, False)
    x.append(response.fun)
    y.append(ret)

# plot
plotting.mean_var_locus(x, y, std, mean, 'Mean Variance Locus (With Risk Free Asset)')
"""

response = optimization.mean_var_portfolio(df_rf, target_returns, N, True, "gurobi")

plotting.mean_var_locus(response[0], response[1], std, mean, 'Mean Variance Locus (No Risk Free Asset)')


"""
# QUESTION 3
N = 49
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)

# the target return becomes a meaningless parameter when dealing with the tangency portfolio
response = optimization.minimize_variance(mean, cov, 0.05, N, True, False, True)

print(response.fun)
print(response.x @ mean)
print(response.x @ cov @ response.x)

fig, ax = plt.subplots()
ax.bar(x=mean.index, height=response.x)
ax.set_title("Weight in Portfolio to Each Asset")
ax.set_xlabel("Assets")
ax.set_ylabel("Weight")
ax.grid()
plt.xticks(rotation=90)
plt.show()

# QUESTION 4
N = 48
cov = df.cov()
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-8, 8, 0.5)
x = []
y = []
for ret in target_returns:
    response = optimization.minimize_variance(mean, cov, ret, N, False, True, False)
    x.append(response.fun)
    y.append(ret)

# plot
plotting.mean_var_locus(x, y, std, mean, 'Mean Variance Locus (No Risk Free Asset & Short Sale Constraint)')

# QUESTION 5
N = 49
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-8, 8, 0.5)
x = []
y = []
for ret in target_returns:
    response = optimization.minimize_variance(mean, cov, ret, N, True, True, False)
    x.append(response.fun)
    y.append(ret)

# plot
plotting.mean_var_locus(x, y, std, mean, 'Mean Variance Locus (With Risk Free Asset & Short Sale Constraint)')

# QUESTION 6
N = 49
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=0)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)

# the target return becomes a meaningless parameter when dealing with the tangency portfolio
response = optimization.minimize_variance(mean, cov, 0.05, N, True, True, True)

print(mean)
print(response.x)
print(response.fun)
print(response.x @ mean)

fig, ax = plt.subplots()
ax.bar(x=mean.index, height=response.x)
ax.set_title("Weight in Portfolio to Each Asset")
ax.set_xlabel("Assets")
ax.set_ylabel("Weight")
ax.grid()
plt.xticks(rotation=90)
plt.show()
"""

# PART B

# 1
N = 49
np.random.seed(101)
nb_bootstraps = 1000
bootstraps = np.zeros(shape=[nb_bootstraps, 60, 49])  # bootstrapped samples

# no short sale constraint
no_rf_mean_var_locus = np.zeros(shape=[nb_bootstraps, len(target_returns), 2])  # no rf rate bootstrapped mean-var locus
rf_mean_var_locus = np.zeros(shape=[nb_bootstraps, len(target_returns), 2])  # rf rate bootstrapped mean-var locus
tangency_portfolio = np.zeros(shape=[nb_bootstraps, 2])  # tangency portfolio bootstrapped mean-var
tangency_portfolio_weights = np.zeros(shape=[nb_bootstraps, 48])  # tangency portfolio bootstrapped weights

for i in tqdm(range(nb_bootstraps)):
    # bootstraps[i, :, :] = df_rf.sample(n=60, replace=True)
    resampled_df = df_rf.sample(n=60, replace=True)
    resampled_df.index = df_rf.index

    mean = resampled_df.iloc[:, 1:].mean(axis=0)
    cov = resampled_df.iloc[:, 1:].cov()  # remove risk-free asset for covariance matrix calculation
    rfr = resampled_df.iloc[1, 1]
    print(np.linalg.det(cov))
    results = []
    for j in range(len(target_returns)):
        ret = float(target_returns[j])

        # A.1
        w, w_rfr = optimization.analytical_mean_var(mean, cov, ret, rfr, 48, False, False)
        no_rf_mean_var_locus[i, j, :] = [np.sqrt(w.T @ cov @ w), mean.T @ w]

        # A.2
        w, w_rfr = optimization.analytical_mean_var(mean, cov, ret, rfr, 48, True, False)
        rf_mean_var_locus[i, j, :] = [np.sqrt(w.T @ cov @ w), mean.T @ w + w_rfr * rfr]

    # A.3
    w, w_rfr = optimization.analytical_mean_var(mean, cov, ret, rfr, 48, False, True)
    tangency_portfolio[i, :] = [np.sqrt(w.T @ cov @ w), mean.T @ w]
    tangency_portfolio_weights[i, :] = w


    # mean = resampled_df.mean(axis=0)
    # std = np.sqrt(np.diag(resampled_df.cov()))



    """
    mean = resampled_df.mean(axis=0)
    std = np.sqrt(np.diag(resampled_df.cov()))
    no_rf_mean_var_locus[i, :, :] = optimization.mean_var_portfolio(resampled_df,
                                                                    target_returns=target_returns,
                                                                    n=N,
                                                                    risk_free_asset=False,
                                                                    optimizer="gurobi")
    """


    # plotting.mean_var_locus(no_rf_mean_var_locus[i, 0, :], no_rf_mean_var_locus[i, 1, :], std, mean, 'Mean Variance Locus (No Risk Free Asset)')


print(no_rf_mean_var_locus[:, 0, 0])
plt.hist(no_rf_mean_var_locus[:, 0, 0])
plt.show()




u_quantile = []
l_quantile = []
for j in range(no_rf_mean_var_locus.shape[1]):
    u_quantile.append(np.quantile(no_rf_mean_var_locus[:, j, 0], 0.95))
    l_quantile.append(np.quantile(no_rf_mean_var_locus[:, j, 0], 0.05))

response = optimization.mean_var_portfolio(df, target_returns, N, False, "gurobi")

plt.plot(response[0], response[1])
plt.plot(l_quantile, target_returns)
plt.plot(u_quantile, target_returns)
plt.show()
