import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

import optimization
import plotting

import warnings

warnings.filterwarnings("ignore")

# import data
data_path = str(Path().absolute()) + "/data_will/48_industry_Portfolios.CSV"
df_global = pd.read_csv(data_path, index_col=0)
df_global.index = pd.to_datetime(df_global.index, format="%Y%m")  # clean the index to be datetime

# constants
start_date = "2019-12-01"  # 5 years worth of data (60 entries)
N = 48
risk_free = 0.4  # taken from Kenneth French Library
df_global = df_global.loc[df_global.index >= start_date, :]  # select last 5 years
# df = df / 100  # convert to percentages
df = df_global.loc[:, ["Whlsl", "Fin  ", "Util ", "Cnstr", "Hlth "]]  # select 5 assets to work with

# QUESTION 1
# first and second moment
cov = df.cov()
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-25, 25, 0.1)

# analytical methodology

inv_sigma = np.linalg.pinv(df.cov())
A = np.ones(5).T @ inv_sigma @ np.ones(5)
B = np.ones(5).T @ inv_sigma @ mean
C = mean.T @ inv_sigma @ mean
delta = A * C - B ** 2

results = []
for ret in target_returns:
    w = (C - ret * B) / delta * inv_sigma @ np.ones(5) + (ret * A - B) / delta * inv_sigma @ mean
    results.append([np.sqrt(w.T @ df.cov() @ w), mean.T @ w])

results = np.array(results)
plotting.mean_var_locus(results[:, 0], results[:, 1], std, mean, 'Mean Variance Locus (No Risk Free Asset)')

response_no_rf, weights = optimization.mean_var_portfolio(df, target_returns, N, False, False, False, "gurobi")
plotting.mean_var_locus(response_no_rf[0], response_no_rf[1], std, mean, 'Mean Variance Locus (No Risk Free Asset)')

# QUESTION 2
N = 5
rf = pd.DataFrame(risk_free, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=1)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1 / N, N)
target_returns = np.arange(-25, 25, 0.01)

response_rf, weights = optimization.mean_var_portfolio(df_rf, target_returns, N, True, False, False, "gurobi")

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
    response = optimization.minimize_variance(mean, cov, ret, N, False, True, False)
    x_no_rf.append(response.fun)
    y_no_rf.append(mean @ response.x)

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

# PART B

# 1
N = 49
np.random.seed(101)
nb_bootstraps = 1000
bootstraps = np.zeros(shape=[nb_bootstraps, 60, 49])  # bootstrapped samples
mean = df.mean(axis=0)
target_returns_no_short = np.linspace(-2, 4, 21)
target_returns_no_short_rf = np.linspace(0.4, 4, 21)

# FIXED VARIANCE BASED CONF INTERVALS
# target_vars_no_short = np.concatenate([np.linspace(-1*x_no_rf[0], -1*min(x_no_rf), 20), np.linspace(min(x_no_rf), max(x_no_rf), 20)])
# no_rf_mean_var_locus_no_short = np.zeros(shape=[nb_bootstraps, len(target_vars_no_short), 2])

# target_vars_no_short_rf = np.concatenate([np.linspace(-1*x_rf[0], -1*min(x_rf), 20), np.linspace(min(x_rf), max(x_rf), 20)])
# rf_mean_var_locus_no_short = np.zeros(shape=[nb_bootstraps, len(target_vars_no_short), 2])

# FIXED MEAN BASED CONFIDENCE INTERVALS
# no short sale constraint
no_rf_mean_var_locus = np.zeros(shape=[nb_bootstraps, len(target_returns), 2])  # no rf rate bootstrapped mean-var locus
rf_mean_var_locus = np.zeros(shape=[nb_bootstraps, len(target_returns), 2])  # rf rate bootstrapped mean-var locus
tangency_portfolio = np.zeros(shape=[nb_bootstraps, 2])  # tangency portfolio bootstrapped mean-var
tangency_portfolio_weights = np.zeros(shape=[nb_bootstraps, 5])  # tangency portfolio bootstrapped weights

# short sale constraint
no_rf_mean_var_locus_no_short = np.zeros(
    shape=[nb_bootstraps, len(target_returns_no_short), 2])  # no rf bootstrapped mean-var locus
rf_mean_var_locus_no_short = np.zeros(
    shape=[nb_bootstraps, len(target_returns_no_short_rf), 2])  # rf bootstrapped mean-var locus
tangency_portfolio_no_short = np.zeros(shape=[nb_bootstraps, 2])  # tangency portfolio bootstrapped mean-var
tangency_portfolio_weights_no_short = np.zeros(shape=[nb_bootstraps, 5])  # tangency portfolio bootstrapped weights

for i in tqdm(range(nb_bootstraps)):
    # bootstraps[i, :, :] = df_rf.sample(n=60, replace=True)
    resampled_df = df_rf.sample(n=60, replace=True)
    resampled_df.index = df_rf.index

    mean = resampled_df.iloc[:, 1:].mean(axis=0)
    cov = resampled_df.iloc[:, 1:].cov()  # remove risk-free asset for covariance matrix calculation
    rfr = resampled_df.iloc[0, 0]
    results = []
    ret = None
    for j in range(len(target_returns)):
        ret = float(target_returns[j])

        # A.1
        w, w_rfr = optimization.analytical_mean_var(mean, cov, ret, rfr, 5, False, False)
        no_rf_mean_var_locus[i, j, :] = [np.sqrt(w.T @ cov @ w), mean.T @ w]

        # A.2
        w, w_rfr = optimization.analytical_mean_var(mean, cov, ret, rfr, 5, True, False)
        rf_mean_var_locus[i, j, :] = [np.sqrt(w.T @ cov @ w), mean.T @ w + w_rfr * rfr]

    # A.3
    w, w_rfr = optimization.analytical_mean_var(mean, cov, ret, rfr, 5, True, True)

    tangency_portfolio[i, :] = [np.sqrt(w.T @ cov @ w), mean.T @ w]
    tangency_portfolio_weights[i, :] = w

    # A.4

    response, weights = optimization.mean_var_portfolio(resampled_df.iloc[:, 1:],
                                                        target_returns=target_returns_no_short,
                                                        n=5,
                                                        risk_free_asset=False,
                                                        long_only=True,
                                                        tangent=False,
                                                        optimizer="gurobi")
    print(np.array(response).T)
    no_rf_mean_var_locus_no_short[i, :, :] = np.array(response).T
    print(no_rf_mean_var_locus_no_short[i, :, :])
    """
    response, weights = optimization.eff_frontier(resampled_df.iloc[:, 1:],
                                                  target_vars=target_vars_no_short,
                                                  n=5,
                                                  risk_free_asset=False,
                                                  long_only=True,
                                                  tangent=False,
                                                  optimizer="gurobi")

    no_rf_mean_var_locus_no_short[i, :, :] = np.array(response).T
    """
    # A.5

    response, weights = optimization.mean_var_portfolio(resampled_df,
                                                        target_returns=target_returns_no_short_rf,
                                                        n=6,
                                                        risk_free_asset=True,
                                                        long_only=True,
                                                        tangent=False,
                                                        optimizer="gurobi")

    rf_mean_var_locus_no_short[i, :, :] = np.array(response).T
    """
    response, weights = optimization.eff_frontier(resampled_df.iloc[:, 1:],
                                                  target_vars=target_vars_no_short_rf,
                                                  n=5,
                                                  risk_free_asset=True,
                                                  long_only=True,
                                                  tangent=False,
                                                  optimizer="gurobi")

    rf_mean_var_locus_no_short[i, :, :] = np.array(response).T
    """

    # A.6
    response = optimization.tangency_portfolio(resampled_df.iloc[:, 1:],
                                               rf=0.4,
                                               long_only=True)

    weights = response.x
    weights = np.array(weights)
    tangency_portfolio_no_short[i, :] = [np.sqrt(weights @ cov @ weights), weights @ mean]
    tangency_portfolio_weights_no_short[i, :] = weights

pd.DataFrame(no_rf_mean_var_locus_no_short[:, 15, :]).to_csv("/Users/william/Desktop/results.csv")

plt.hist(no_rf_mean_var_locus[:, 0, 0])
plt.show()

# no risk-free rate long-short portfolio
plotting.confidence_bands(df,
                          no_rf_mean_var_locus,
                          target_returns,
                          5,
                          "95% Confidence Interval of Mean Variance Locus (No Risk Free Asset)",
                          False,
                          False)

# risk-free rate long-short portfolio
plotting.confidence_bands(df_rf,
                          rf_mean_var_locus,
                          target_returns,
                          6,
                          "95% Confidence Interval of Mean Variance Locus (With Risk Free Asset)",
                          True,
                          False)

# scatterplot of tangency portfolio standard deviation against expected return
# regression line
x = tangency_portfolio[:, 0]
y = tangency_portfolio[:, 1]
# m, b = np.polyfit(x, y, 1)

fig, ax = plt.subplots()
ax.scatter(x, y)
# ax.plot(x, m * x + b, linewidth=2, color="black")
ax.set(xlabel='Standard Deviation (%)', ylabel='Expected Return (%)',
       title="Bootstrapped Tangency Portfolio Expected Return & Standard Deviation ScatterPlot")
ax.grid()
plt.show()

x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots()
ax.scatter(x[x < 100], y[x < 100])
# ax.plot(x, m * x + b, linewidth=2, color="black")
ax.set(xlabel='Standard Deviation (%)', ylabel='Expected Return (%)',
       title="Bootstrapped Tangency Portfolio Expected Return & Standard Deviation Truncated at 100% ScatterPlot")
ax.grid()
plt.show()



fig, ax = plt.subplots()
ax.plot(np.arange(1, 1001), (y-risk_free)/x)
ax.set(xlabel='Index', ylabel='Sharpe Ratio',
       title="Tangency Portfolio Sharpe Ratio per Bootstrapped Sample (No Short Sale Constraint)")
ax.grid()
plt.show()

# box plot of tangency portfolio weights
tangency_portfolio_weights = pd.DataFrame(tangency_portfolio_weights,
                                          columns=["Whlsl", "Fin  ", "Util ", "Cnstr", "Hlth "])
plt.boxplot(tangency_portfolio_weights, labels=["Whlsl", "Fin  ", "Util ", "Cnstr", "Hlth "], showmeans=True)
plt.ylabel("Weights (%)")
plt.xlabel("Industries")
plt.title("Weights in Tangency Portfolio")
plt.show()

# no risk-free rate long-only
plotting.confidence_bands(df,
                          no_rf_mean_var_locus_no_short,
                          target_returns_no_short,
                          5,
                          "95% Confidence Interval of Mean Variance Locus (No Risk Free Asset & Long Only)",
                          False,
                          True)
"""
plotting.confidence_bands_max_ret(df,
                                  no_rf_mean_var_locus_no_short,
                                  target_vars_no_short,
                                  target_returns,
                                  5,
                                  "95% Confidence Interval of Mean Variance Locus (No risk-free asset & Long Only)",
                                  False,
                                  True,
                                  upper_portion=True)

plotting.confidence_bands_max_ret(df,
                                  no_rf_mean_var_locus_no_short,
                                  target_vars_no_short,
                                  target_returns,
                                  5,
                                  "95% Confidence Interval of Mean Variance Locus (No risk-free asset & Long Only)",
                                  False,
                                  True,
                                  upper_portion=False)
"""
# risk-free rate long-only

plotting.confidence_bands(df_rf,
                          rf_mean_var_locus_no_short,
                          target_returns_no_short_rf,
                          6,
                          "95% Confidence Interval of Mean Variance Locus (With Risk Free Asset & Long Only)",
                          True,
                          True)
"""
plotting.confidence_bands_max_ret(df,
                                  rf_mean_var_locus_no_short,
                                  target_vars_no_short_rf,
                                  target_returns,
                                  5,
                                  "95% Confidence Interval of Mean Variance Locus (No risk-free asset & Long Only)",
                                  True,
                                  True,
                                  upper_portion=True)

plotting.confidence_bands_max_ret(df,
                                  rf_mean_var_locus_no_short,
                                  target_vars_no_short_rf,
                                  target_returns,
                                  5,
                                  "95% Confidence Interval of Mean Variance Locus (No risk-free asset & Long Only)",
                                  True,
                                  True,
                                  upper_portion=False)
"""

# scatterplot of tangency portfolio standard deviation against expected return
# regression line
x = tangency_portfolio_no_short[:, 0]
y = tangency_portfolio_no_short[:, 1]
# m, b = np.polyfit(x, y, 1)

fig, ax = plt.subplots()
ax.scatter(x, y)
# ax.plot(x, m * x + b, linewidth=2, color="black")
ax.set(xlabel='Standard Deviation (%)', ylabel='Expected Return (%)',
       title="Bootstrapped Tangency Portfolio Expected Return & Standard Deviation ScatterPlot")
ax.grid()
plt.show()


fig, ax = plt.subplots()
ax.plot(np.arange(1, 1001), (y-risk_free)/x)
ax.set(xlabel='Index', ylabel='Sharpe Ratio',
       title="Tangency Portfolio Sharpe Ratio per Bootstrapped Sample (Long Only)")
ax.grid()
plt.show()

# box plot of tangency portfolio weights
tangency_portfolio_weights_no_short = pd.DataFrame(tangency_portfolio_weights_no_short,
                                                   columns=["Whlsl", "Fin  ", "Util ", "Cnstr", "Hlth "])
plt.boxplot(tangency_portfolio_weights_no_short, labels=["Whlsl", "Fin  ", "Util ", "Cnstr", "Hlth "], showmeans=True)
plt.title("Weights in Long Only Tangency Portfolio")
plt.xlabel("Industries")
plt.ylabel("Weights (%)")
plt.show()
