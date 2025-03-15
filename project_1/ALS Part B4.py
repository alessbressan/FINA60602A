import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import betainc, gamma
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from adjustText import adjust_text
import warnings
from pathlib import Path
import datetime
import os




# DATA PREPARATION______________________________________________________________________________________________________

warnings.filterwarnings("ignore", category=FutureWarning)

# Load and read industry data from csv file

# file_path = "C:/Users/Anjan Nanan/Desktop/PM Assignment 1/Raw Data/Industry Data.csv"
file_path = str(Path().absolute()) + "/data_will/48_industry_Portfolios.CSV"
df = pd.read_csv(file_path, index_col=0)

# Convert index column into a date format
df.index = pd.to_datetime(df.index, format='%Y%m')

# Select data from the last five years
df = df.loc["2019-12-01":]

num_rows = len(df)

# Number of industries -> N = 48
N = len(df.columns)

# Risk-free rate tahed from FF 3-factor model
rf = 0.4



# STEP 0________________________________________________________________________________________________________________

# Form 1000 subpools of randomly picked N-assets

industries = list(df.axes[1])
N_subpool = 1000
subpool_size = 10
N =48

subpools = {}

random.seed(50)
# Randomly pick N-asset for each subpool
# Create subpools
for i in range(N_subpool):
    subpool = random.sample(industries, subpool_size)
    subpools[i] = subpool



for i in range(5):
    print(f"Subpool {i + 1}: {subpools[i]}")

index_r =[]

# Print the subpools with their indices
for index, subpool in subpools.items():
    #print(f"Subpool {index}: {subpool}")
    index_r.append(index)


print(index_r)
#subpools = df.iloc[:, index_r]



# Calculate excess return and covariance matrix
mean = df.mean()
ex_ret = df.mean() - rf
cov = df.cov()
inv_cov = np.linalg.inv(cov)
std = np.sqrt(np.diag(cov))


# print(df[subpools[1]].mean())

# Function to calculate Squared Sharpe Ratio - theta_adj


def theta_adj(T: float, N: float, ex_ret, inv_cov):

    theta_s = ex_ret.T @ inv_cov @ ex_ret


    # Inputs to calculate the Incomplete beta function
    a = N / 2
    b = (T - N) / 2
    x = theta_s / (1 + theta_s)
    beta_inc = betainc(a, b, x) * gamma(a) * gamma(b) / gamma(a+b)

    # The theta_adj function

    first_eqn = (((T - N - 2) * theta_s) - N) / T
    second_eqn = ((2 * theta_s ** (N / 2)) * (1 + theta_s) ** (-(T - 2) / 2)) / (T * beta_inc)

    theta_adj = first_eqn + second_eqn

    return theta_adj


# To calculate the theta_adj

max_sharpe = []

def sq_max_sharpe(T , N):
  for i in range(N_subpool):
        mean = df[subpools[i]].mean()

        ex_ret = df[subpools[i]].mean() - rf

        cov = df[subpools[i]].cov()

        inv_cov = np.linalg.inv(cov)

        sq_max_sharpe = theta_adj(T, N, ex_ret, inv_cov)

        max_sharpe.append((i, np.sqrt(sq_max_sharpe)))

  return max_sharpe


res = sq_max_sharpe(60, 10)


# To select subpool that corresponds to the 95% quantile

dff = pd.DataFrame()

dff['Index'] = [sublist[0] for sublist in max_sharpe]
dff['Max_Sharpe'] = [sublist[1] for sublist in max_sharpe]
print(dff.head(187))

quant_95 = np.percentile(dff['Max_Sharpe'], 95)
closest_index = (dff['Max_Sharpe'] - quant_95).abs().idxmin()
print(closest_index)
print(quant_95)


# calculate response r_c from our squared sharpe ratio
# we set the risk constraint equal to the stdev of the tangency portfolio
sigma = 3
r_c = ((1 + quant_95 ** 2) / quant_95) * sigma

print(r_c)


# STEP 2 FOR BEST LAMBDA

# we first find the OLS solution
X = df[subpools[closest_index]].values
y = np.full(X.shape[0], r_c)
w_ols = np.linalg.inv(X.T @ X) @ X.T @ y

kf = KFold(n_splits=10, shuffle=True, random_state=42)

zetas = np.zeros(shape=10)
for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    best_diff = 10000
    # consider each possible lambda value to find the one which matches our desired risk metric
    for j in range(101):
        zeta = j / 100
        w_lasso = zeta * w_ols
        sigma_test = np.sqrt(w_lasso @ np.cov(X[test_idx].T) @ w_lasso)

        if abs(sigma_test - sigma) < best_diff:
            zetas[i] = zeta
            best_diff = abs(sigma_test - sigma)

print(zetas)
lambda_reg = np.mean(zetas)

# STEP 3
def maxser_weights(data, response, lambda_val):
    from scipy.optimize import minimize

    X = data.values
    y = np.full(X.shape[0], response)

    def loss_fn(weights):
        return np.mean((y - X @ weights) ** 2)

    constraints = ({'type': 'ineq', 'fun': lambda w: lambda_val - np.sum(np.abs(w))})
    # bounds = [(0, None)] * data.shape[1]
    bounds = [(None, None)] * data.shape[1]

    result = minimize(loss_fn, np.ones(data.shape[1]) / data.shape[1], bounds=bounds, constraints=constraints, method="SLSQP")
    return result.x


weights = maxser_weights(df[subpools[closest_index]], r_c, lambda_reg)

# Final portfolio weights
portfolio_weights = pd.Series(weights, index=df[subpools[closest_index]].columns)
print("MAXSER Portfolio Weights:")
print(portfolio_weights)

#sum portfolio weights
print("Sum of Portfolio Weights:")
print(portfolio_weights.sum())

print(portfolio_weights @ (df[subpools[closest_index]].mean() - rf))
print(np.sqrt(portfolio_weights @ (df[subpools[closest_index]].cov() @ portfolio_weights)))


plt.figure(figsize=(10, 6))
plt.bar(x=portfolio_weights.index, height=portfolio_weights.values)
plt.xticks(rotation=45)
plt.title("MAXSER Portfolio Weights")
plt.xlabel("Portfolio")
plt.ylabel("Weight")
plt.show()
