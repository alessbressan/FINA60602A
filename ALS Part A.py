import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.optimize import minimize
from adjustText import adjust_text
import warnings
import datetime
import os





# DATA PREPARATION______________________________________________________________________________________________________

warnings.filterwarnings("ignore", category=FutureWarning)

# Load and read industry data from csv file
path = os.getcwd()
file_path = path + "\Industry Data.csv"
df = pd.read_csv(file_path, index_col = 0)

# Convert index column into a date format
df.index = pd.to_datetime(df.index, format = '%Y%m')

# Select data from the last five years
df = df.loc["2019-12-01":]

num_rows = len(df)

# Number of industries -> N = 48
N = len(df.columns)



# PART A
# QUESTION 1____________________________________________________________________________________________________________



# Graph the "mean-variance locus" (without the risk-free asset) of the 48 industry portfolios.
# Specify each industry portfolio in the chart


# Compute mean, standard deviation and covariance matrix
mean = df.mean()
cov = df.cov()
std = np.sqrt(np.diag(cov))


# Objective is to minimize standard deviation for a given set of returns
def port_std(weights, sigma):
    return np.sqrt(weights.T @ sigma @ weights)


# Initial weights and target returns
initial_w = np.repeat(1/N, N)
target_returns = np.arange(-6, 6, 0.25)


# Minimize volatility
minimized_std = []

for ret in target_returns:
     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: x @ mean - ret})

     result = minimize(fun = port_std, x0=initial_w, args = (cov),
                       method = 'SLSQP', constraints = constraints, bounds = None, options = {'maxiter': 500})

     if result.success:
         minimized_std.append(result['fun'])


# Plot Efficient Frontier

# Add industry labels
names = list(df.axes[1])
texts = []
for i in range(len(names)):
    texts.append(plt.text(std[i], mean[i], names[i], fontsize=10))


# Plot data with color and arrow display
colors = np.arange(0, N, 1)
plt.plot(minimized_std, target_returns)
plt.scatter(std, mean, marker='o', c = colors, cmap = 'plasma')
adjust_text(texts, arrowprops = dict(arrowstyle = "->", color = "grey", lw = 0.5))
plt.title('Mean-Variance Locus (without the risk-free asset)')
plt.xlabel('Expected Volatility (%)')
plt.ylabel('Expected Returns (%)')

plt.grid()
plt.show()





# QUESTION 2____________________________________________________________________________________________________________

# Graph the "mean-variance locus" (with the risk-free asset) of these 48 industry portfolios
# Specify each industry portfolio in the chart
# Explain how the mean-variance locus has changed with the risk-free asset

# Update N to include the risk-free asset
N_rf = 49

# Add a column for the risk-free asset
df_rf = df.assign(Risk_free_asset = np.repeat(1,num_rows))


mean_rf = df_rf.mean()
cov = df.cov()
std_rf = np.sqrt(np.diag(df_rf.cov()))


def port_std_rf(weights, sigma):
    return np.sqrt(weights[:-1].T @ sigma @ weights[:-1])


# Initial weights and target returns
initial_w_rf = np.repeat(1/N_rf, N_rf)
target_returns_rf = np.arange(-6, 6, 0.25)


# Minimize volatility
minimized_std_rf = []

for ret in target_returns_rf:
     constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: x @ mean_rf - ret })



     result = minimize(fun = port_std_rf, x0 = initial_w_rf, args = (cov),
                       method = 'SLSQP', constraints = constraints, bounds = None, options={'maxiter': 500})

     if result.success:
         minimized_std_rf.append(result['fun'])


# Add industry labels
names = list(df_rf.axes[1])
texts = []
for i in range(len(names)):
    texts.append(plt.text(std_rf[i], mean_rf[i], names[i], fontsize=10))


# Plot data with color and arrow display
colors = np.arange(0, N_rf, 1)
plt.plot(minimized_std_rf,target_returns)
plt.scatter(std_rf, mean_rf, marker='o', c = colors, cmap = 'plasma')
adjust_text(texts,arrowprops = dict(arrowstyle = "->", color = "grey", lw  = 0.5))
plt.title('Mean-Variance Locus (with the risk-free asset)')
plt.xlabel('Expected Volatility (%)')
plt.ylabel('Expected Returns (%)')
plt.grid()
plt.show()




# QUESTION 4____________________________________________________________________________________________________________

# Graph the "mean-variance locus" (without the risk-free asset) with the short-sale constraints
# Specify each industry portfolio in the chart



# Long-only portfolio
bounds = []
for n in range(N):
    bounds.append((0, None))


minimized_std = []
returns = []

for ret in target_returns:
     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: x @ mean - ret })

     result = minimize(fun = port_std, x0 = initial_w, args = (cov),
                       method = 'SLSQP', constraints = constraints, bounds = bounds, options = {'maxiter': 500})

     if result.success:
         minimized_std.append(result['fun'])
         returns.append(ret)


# Add industry labels
names = list(df.axes[1])
texts = []
for i in range(len(names)):
    texts.append(plt.text(std[i], mean[i], names[i], fontsize=10))


# Plot data with color and arrow display
colors = np.arange(0, N, 1)
plt.plot(minimized_std, returns)
plt.scatter(std, mean, marker='o', c = colors, cmap = 'plasma')
plt.title('Mean-Variance Locus (without the risk-free asset) - short sale constraints')
adjust_text(texts,arrowprops = dict(arrowstyle = "->", color = "grey", lw = 0.5))
plt.xlabel('Expected Volatility (%)')
plt.ylabel('Expected Returns (%)')
plt.grid()
plt.show()



# QUESTION 5____________________________________________________________________________________________________________

# Graph the "mean-variance locus" (with the risk-free asset) with the short-sale constraints
# Specify each industry portfolio in the chart

# Long-only portfolio
bounds = []
for n in range(N_rf):
    bounds.append((0, None))


minimized_std_rf = []
returns_rf = []

for ret in target_returns:
     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: x @ mean_rf - ret })

     result = minimize(fun = port_std_rf, x0 = initial_w_rf, args = (cov),
                       method = 'SLSQP', constraints = constraints, bounds = bounds, options = {'maxiter': 500})

     if result.success:
         minimized_std_rf.append(result['fun'])
         returns_rf.append(ret)


# Add industry labels
names = list(df_rf.axes[1])
texts = []
for i in range(len(names)):
    texts.append(plt.text(std_rf[i], mean_rf[i], names[i], fontsize=10))


# Plot data with color and arrow display
colors = np.arange(0, N_rf, 1)
plt.plot(minimized_std_rf, returns_rf)
plt.scatter(std_rf, mean_rf, marker='o', c = colors, cmap = 'plasma')
adjust_text(texts,arrowprops = dict(arrowstyle = "->", color = "grey", lw = 0.5))
plt.title('Mean-Variance Locus (with the risk-free asset) - short sale constraints')
plt.xlabel('Expected Volatility (%)')
plt.ylabel('Expected Returns (%)')
plt.grid()
plt.show()