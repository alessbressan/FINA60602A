import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import betainc, gamma
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
import warnings
import datetime





# DATA PREPARATION______________________________________________________________________________________________________

warnings.filterwarnings("ignore", category=UserWarning)

# Load and read industry data from csv file

file_path = "C:/Users/Anjan Nanan/Desktop/PM Assignment 1/Raw Data/Industry Data.csv"
df = pd.read_csv(file_path, index_col = 0)

# Convert index column into a date format
df.index = pd.to_datetime(df.index, format = '%Y%m')

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
    beta_inc = betainc(a , b , x) * gamma(a) * gamma(b) / gamma(a+b)

    # The theta_adj function

    first_eqn = (((T - N - 2) * theta_s) - N ) / T
    second_eqn = ((2 * (theta_s) ** (N / 2)) * (1 + theta_s) ** (-(T - 2) / 2)) / (T * beta_inc)

    theta_adj = first_eqn + second_eqn

    return theta_adj


# To calculate the max sharpe

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

print(max_sharpe)
res = sq_max_sharpe(60, 10)


# To select subpool that corresponds to the 95% quantile

dff = pd.DataFrame()

dff['Index'] = [sublist[0] for sublist in max_sharpe]
dff['Max_Sharpe'] = [sublist[1] for sublist in max_sharpe]
print(dff.head(187))

quant_95 = np.percentile(dff['Max_Sharpe'], 95)
closest_index = (dff['Max_Sharpe'] - quant_95).abs().idxmin()
print("Subpool index: ", closest_index)
print("95th quantile: ", quant_95)



# STEP 1________________________________________________________________________________________________________________

# Estimate the square of the maximum Sharpe ratio by theta_adj, and compute the response r_c

sigma = 1
r_c = sigma * ((1 + quant_95 ** 2) / quant_95)
print("R_c response: ", r_c)



# STEP 2________________________________________________________________________________________________________________

# 1. Need to get data for last five years for chosen subpool

sample = df[subpools[closest_index]].values
len_sample = len(sample)
zetas = np.linspace(0, 1, 50)



# 2. Need to create an array with the r_c response
rc_array = np.repeat(r_c, len_sample)


# 3. Need to to split the sample in part 1 into 10 groups - use Kfold
# Note: For each validation set, the training set is taken to be the rest of the observations in the sample
X = sample
y = rc_array

kf = KFold(n_splits=10, shuffle=True, random_state=42)


# 4. For each training set i, get the whole solution for zeta between 0 and 1 where zeta = ||w|| / ||w_ols||
# Perform K-Fold Cross-Validation

best_min = float('inf')
best_zetas = []

for zeta in zetas:
    for train_index, test_index in kf.split(X):


        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        lasso = Lasso(alpha = zeta, fit_intercept=False)  # 10-fold cross-validation
        lasso.fit(X_train, y_train)
        w = lasso.coef_

        sigma_test = np.sqrt(w @ np.cov(X_test.T) @ w)

        min_risk = abs(sigma_test - sigma)

        # Calculate the zeta corresponding to the portfolio that minimizes the diff b/n the risk computed using the validation set and the given risk constraint
        if min_risk < best_min:
           best_min = min_risk
           best_zetas.append(best_min)


# The ultimate zeta is taken to be the average of the zetas (1,...,10)
ultimate_zeta = np.mean(best_zetas)
print("Ultimate Zeta: ",ultimate_zeta)


# Using beta_ols solution (X'X)^-1(X'Y) to get w_ols
X = sample
Y = np.repeat(r_c, len(X))

w_ols = np.linalg.inv(X.T @ X) @ X.T @ Y
#print(w_ols)



# STEP 3________________________________________________________________________________________________________________

# Set lambda_hat to be equal to ultimate_zeta * ||w_ols||1
lambda_hat = ultimate_zeta * np.sum(abs(w_ols))
print("Lambda_hat: ", lambda_hat)


# Use Lasso-type approach to get optimal weights
lasso = Lasso(alpha = lambda_hat, fit_intercept=False)
lasso.fit(X, Y)
weight_opt = lasso.coef_

colors = ['b' if i >= 0 else 'r' for i in weight_opt]

plt.figure(figsize=(10, 6))
plt.bar(x = df[subpools[closest_index]].columns, height = weight_opt, color=colors)
plt.xticks(rotation=45)
plt.title("MAXSER Portfolio Weights")
plt.xlabel("Portfolio")
plt.ylabel("Weight")
plt.show()






