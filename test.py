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


if __name__ == "__main__":
    # import data
    data_path = str(Path().absolute()) + "/clean_data/48_Industry_Portfolios.CSV"
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

    # global mean variance portfolio
    init_w = np.repeat(1/N, N)
    ret = 2
    response = optimize_portfolio_gurobi(cov, mean, ret, False, False, False)
    print(response)
    # plot
    # plotting.mean_var_locus(x, y, std, mean, 'Mean Variance Locus (No Risk Free Asset)')
