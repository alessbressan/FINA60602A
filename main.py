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


# QUESTION 1
# first and second moment
cov = df.cov()
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
init_w = np.repeat(1/N, N)
target_returns = np.arange(-8, 8, 0.5)
x = []
y = []
for ret in target_returns:
    response = optimization.minimize_variance(mean, cov, ret, N, False, False, False)
    x.append(response.fun)
    y.append(ret)

# plot
plotting.mean_var_locus(x, y, std, mean, 'Mean Variance Locus (No Risk Free Asset)')


# QUESTION 2
N = 49
rf = pd.DataFrame(1, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=0)


# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1/N, N)
target_returns = np.arange(-8, 8, 0.5)
x = []
y = []
for ret in target_returns:
    response = optimization.minimize_variance(mean, cov, ret, N, True, False, False)
    x.append(response.fun)
    y.append(ret)

# plot
plotting.mean_var_locus(x, y, std, mean, 'Mean Variance Locus (With Risk Free Asset)')


# QUESTION 3
N = 49
rf = pd.DataFrame(1, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=0)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1/N, N)

# the target return becomes a meaningless parameter when dealing with the tangency portfolio
response = optimization.minimize_variance(mean, cov, 0.05, N, True, False, True)

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


# QUESTION 4
N = 48
cov = df.cov()
mean = df.mean(axis=0)
std = np.sqrt(np.diag(cov))

# global mean variance portfolio
init_w = np.repeat(1/N, N)
target_returns = np.arange(min(mean), max(mean), 0.05)
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
rf = pd.DataFrame(1, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=0)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1/N, N)
target_returns = np.arange(min(mean), max(mean), 0.05)
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
rf = pd.DataFrame(1, index=df.index, columns=["risk-free"])
df_rf = pd.concat([rf, df], axis=0)

# cov = df.cov()
mean = df_rf.mean(axis=0)
std = np.sqrt(np.diag(df_rf.cov()))

# global mean variance portfolio
init_w = np.repeat(1/N, N)

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
