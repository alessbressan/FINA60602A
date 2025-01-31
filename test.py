import pandas as pd
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from adjustText import adjust_text
from pathlib import Path
import warnings
import gurobipy as gp
from gurobipy import GRB

import optimization
import plotting

warnings.filterwarnings("ignore", category=FutureWarning)



def optimize_portfolio_gurobi(sigma, mean, ret, long_only=True, tangent=False, risk_free_asset=False):
    """
    Portfolio optimization using Gurobi.

    Args:
        sigma (np.ndarray): Covariance matrix.
        mean (np.ndarray): Expected returns.
        ret (float): Target return.
        long_only (bool): If True, weights must be non-negative.
        tangent (bool): If True, solves for tangency portfolio.
        risk_free_asset (bool): If True, includes risk-free asset.

    Returns:
        np.ndarray: Optimal weights.
    """
    n = len(mean)  # Number of assets
    model = gp.Model("PortfolioOptimization")

    # Add variables for weights
    if long_only:
        weights = model.addVars(n, lb=0, ub=1, name="weights")  # long-only bounds
    else:
        weights = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="weights")  # no bounds

    # Risk-free asset handling for tangency portfolio
    if tangent and risk_free_asset:
        weights[0].lb = 0
        weights[0].ub = 0

    # Add constraints
    model.addConstr(weights.sum() == 1, "BudgetConstraint")  # weights must sum to 1
    model.addConstr(weights @ mean == ret, "TargetReturn")  # target return

    # Define the quadratic objective: minimize x^T Sigma x
    model.setObjective(weights.T @ sigma @ weights, GRB.MINIMIZE)

    # Solve the model
    model.setParam("OutputFlag", 0)  # Turn off Gurobi output for cleaner output
    model.optimize()

    # Check if optimization was successful
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimization failed: {model.status}")

    # Extract optimal weights
    optimal_weights = np.array([weights[i].X for i in range(n)])

    return optimal_weights

def max_sharpe_ratio(df: pd.DataFrame, exp_return: float) -> tuple:
    """ 
    Computes the maximum sharpe ratio for a given return

    Inputs
        df: dataframe of monthly returns, each colum is a different asset
        exp_return: expected return for the sharpe ratio
    
    Outputs
        SR: max Sharpe Ratio for the return
        x: weights of the assets
        result: dataframe of positions 
    """
    cov = df.cov()
    mean = df.mean(axis=0)

    K = 5
    l = 5 #max position size (500%)

    # Create an empty optimization model
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads=8

    # Add variables: x[i] denotes the proportion of capital invested in stock i
    x = m.addMVar(len(mean), lb=0, ub=np.inf, name="x")
    b = m.addMVar(len(mean), vtype=gp.GRB.BINARY, name="b")

    # Budget constraint: all investments sum up to 1
    m.addConstr(x.sum() == 1, name="Budget_Constraint")
    m.addConstr(x <= l*b, name= "Long_Indicator")
    m.addConstr(b.sum() <= K, name="Cardinality")
    m.addConstr(x.T @ mean.to_numpy() >= exp_return , name="Target_Return")

    # Minimize variance
    m.setObjective(x.T @ cov.to_numpy() @ x, gp.GRB.MINIMIZE)

    m.optimize()

    var = x.X @ cov.to_numpy() @ x.X
    rets = mean @ x.X
    SR = rets/np.sqrt(var)

    positions = pd.Series(name="Position", data= x.X, index= mean.index)
    index = positions[abs(positions) > 1e-5].index
    result = pd.DataFrame({'mean' : df[index].mean(),
                            'var' : df[index].var()})
    
    return SR, x, result


if __name__ == "__main__":
    # import data
    data_path = str(Path().absolute()) + "/data_will/48_Industry_Portfolios.CSV"
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y%m")  # clean the index to be datetime

    # constants
    start_date = "2020-01-01"
    N = 48
    df = df.loc[df.index >= start_date, :]  # select last 5 years
    # df = df / 100  # convert to percentages

    # first and second moment
    cov = df.cov()
    mean = df.mean(axis=0).to_numpy()
    std = np.sqrt(np.diag(cov))

    SR, _, _ = max_sharpe_ratio(df, 0.03)
    print(SR)