import numpy as np
import pandas as pd
from scipy import optimize
from statsmodels.stats.correlation_tools import cov_nearest
import gurobipy as gp


def portfolio_std(w: np.array, sigma: np.array, risk_free_asset: bool):
    """
    computes variance of portfolio
    :param w: weights of each asset in portfolio
    :param sigma: variance-covariance matrix of portfolio assets
    :param risk_free_asset: boolean determining whether there is a risk-free asset
    :return: standard deviation of portfolio
    """

    if risk_free_asset:
        return np.sqrt(w[1:].T @ sigma @ w[1:])
    else:
        return np.sqrt(w.T @ sigma @ w)


def neg_sharpe_ratio(w: np.array, sigma: np.array, mean: np.array, rfr: float):
    """
    computes sharpe ratio of portfolio
    :param w: weights of each asset in portfolio (no risk free-asset)
    :param sigma: variance-covariance matrix of portfolio assets (no risk-free asset)
    :param mean: average returns of assets (no risk-free asset)
    :param rfr: risk-free rate
    :return: sharpe of portfolio
    """

    return -1 * (w @ mean - rfr) / np.sqrt(w @ sigma @ w)


def minimize_variance(mean: np.array, sigma: np.array, ret: float, n: int, risk_free_asset: bool, long_only: bool,
                      tangent: bool):
    """
    minimizes variance of portfolio using sequential least squares programming and returns minimized std
    :param mean: mean returns of each asset in portfolio
    :param sigma: variance-covariance matrix of portfolio of assets
    :param ret: the target return for the portfolio against which we minimize variance
    :param n: number of risky assets in portfolio (used to set equal weight portfolio as initial guess)
    :param risk_free_asset: boolean determining whether there is a risk-free asset
    :param long_only: boolean adding a no short sale constraint
    :param tangent: boolean to determine whether we want the tangency portfolio
    :return: standard deviation of portfolio
    """
    init_w = np.repeat(1 / n, n)

    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},  # weights must sum to 1
            {"type": "eq", "fun": lambda x: x @ mean - ret})  # portfolio must have the target return

    if tangent and risk_free_asset:
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},  # weights must sum to 1
                {"type": "eq", "fun": lambda x: x[0] - 0})  # portfolio must have 0% holdings in risk-free asset

    elif tangent and not risk_free_asset:
        ValueError("Set risk-free asset to True to compute tangency portfolio")

    if long_only:
        bounds = []
        for i in range(n):
            bounds.append((0, None))

        bounds = tuple(bounds)

        response = optimize.minimize(portfolio_std, init_w,
                                     (sigma, risk_free_asset),
                                     constraints=cons,
                                     bounds=bounds,
                                     method="SLSQP")

    else:
        response = optimize.minimize(portfolio_std, init_w,
                                     (sigma, risk_free_asset),
                                     constraints=cons,
                                     method="SLSQP")

    return response


def minimize_variance_gurobi(mean: np.array, sigma: np.array, ret: float, risk_free_asset: bool, long_only: bool,
                             tangent: bool):
    """
    minimizes variance of portfolio using sequential least squares programming and returns minimized std
    :param mean: mean returns of each asset in portfolio
    :param sigma: variance-covariance matrix of portfolio of assets
    :param ret: the target return for the portfolio against which we minimize variance
    :param risk_free_asset: boolean determining whether there is a risk-free asset
    :param long_only: boolean adding a no short sale constraint
    :param tangent: boolean to determine whether we want the tangency portfolio
    :return: standard deviation of portfolio
    """

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.LogToConsole = 0

    if long_only:
        w = m.addMVar(len(mean), lb=0, ub=1, name="weights")
        m.addConstr(w.sum() == 1, name="Budget_Constraint")

        if (tangent and risk_free_asset):
            m.addConstr(w[0] == 0, name="Tangency_Constraint")

        elif tangent and not risk_free_asset:
            ValueError("Set risk-free asset to True to compute tangency portfolio")

    else:
        w = m.addMVar(len(mean), lb=-np.inf, ub=np.inf, name="weights")
        m.addConstr(w.sum() == 1, name="Budget_Constraint")

        if tangent and risk_free_asset:
            m.addConstr(w[0] == 0, name="Tangency_Constraint")

        elif tangent and not risk_free_asset:
            ValueError("Set risk-free asset to True to compute tangency portfolio")

    # the return constraint is useless when considering the tangent portfolio
    # this is because the tangent portfolio already requires specific return by forcing risk-free weight to zero
    if not tangent:
        m.addConstr(mean.to_numpy() @ w == ret, name="Min_Return")

    m.setObjective(w @ sigma.to_numpy() @ w, gp.GRB.MINIMIZE)
    m.optimize()

    return m


def mean_var_portfolio(df: pd.DataFrame,
                       target_returns: np.array,
                       n: int,
                       risk_free_asset: bool,
                       long_only: bool,
                       tangent: bool, optimizer: str):
    """
    computes the return and standard deviation of each mean variance portfolio for a given expected return
    :param df: dataframe containing the returns of each industry (rows are dates, columns are industries)
    :param target_returns: list of expected returns we construct portfolios to target to build locus
    :param n: number of industries
    :param risk_free_asset: boolean determining whether we're building portfolio with or without risk-free asset
    :param long_only: boolean determining whether we're building long only portfolio or not
    :param tangent: boolean determining whether we're tangent portfolio or not
    :param optimizer: written 'scipy' or 'gurobi'. If gurobi, uses gurobi, else uses scipy
    :return:
    """
    # first and second moment
    cov = df.cov()
    mean = df.mean(axis=0)

    x = np.full(shape=len(target_returns), fill_value=None)  # std
    y = np.full(shape=len(target_returns), fill_value=None)  # target return
    w = np.full(shape=len(target_returns), fill_value=None)  # weights
    if optimizer == "scipy":
        # account for removal of risk-free asset in var-covar matrix
        if risk_free_asset:
            cov = df.iloc[:, 1:].cov()

        for j in range(len(target_returns)):
            ret = target_returns[j]
            response = minimize_variance(mean, cov, ret, n, risk_free_asset, long_only, tangent)
            x[j] = response.fun
            y[j] = mean @ response.x
            w[j] = response.x

    elif optimizer == "gurobi":

        for j in range(len(target_returns)):
            ret = target_returns[j]
            response = minimize_variance_gurobi(mean, cov, ret, risk_free_asset, long_only, tangent)

            try:
                x[j] = np.sqrt(response.ObjVal)
                y[j] = mean @ response.X
                w[j] = response.X
            except AttributeError:
                x[j] = None
                y[j] = None
                w[j] = [None] * n
            except gp._exception.GurobiError:
                w[j] = [None] * n


    else:
        ValueError("Use Gurobi or Scipy as optimizer input")

    return [x, y], w


def analytical_mean_var(mean: np.array, sigma: np.array, ret: float, rfr: float, n: int, risk_free_asset: bool,
                        tangent: bool):
    """
    minimizes variance of portfolio using analytical formulas
    :param mean: mean returns of each asset in portfolio
    :param sigma: variance-covariance matrix of portfolio of assets
    :param rfr: risk-free rate
    :param n: number of industries
    :param ret: the target return for the portfolio against which we minimize variance
    :param risk_free_asset: boolean determining whether there is a risk-free asset
    :param tangent: boolean to determine whether we want the tangency portfolio
    :return: standard deviation of portfolio
    """
    # use pseudo inverse due to numerical instability of covariance matrix
    # indeed the numerical instability is due to repeated samples causing the matrix to possess multi collinear columns
    # and thus to have a lower rank. This causes the determinant of the vcv matrix to approach zero and leads
    # to numerical instability. For the vcv to be invertible, the underlying matrices rank must be larger than
    # the number of columns, which is almost guaranteed to be false if we use 60 rows with resampling
    inv_sigma = np.linalg.pinv(sigma)
    a = np.ones(n).T @ inv_sigma @ np.ones(n)
    b = np.ones(n).T @ inv_sigma @ mean
    c = mean.T @ inv_sigma @ mean
    delta = a * c - b ** 2

    w = None
    w_rfr = None
    if (not risk_free_asset) and tangent:
        ValueError("Set risk-free asset to True to compute tangency portfolio")

    # weights with risk-free asset
    elif risk_free_asset and (not tangent):
        sharpe_sq = c - 2 * b * rfr + a * rfr ** 2
        w = ((ret - rfr) * inv_sigma @ (mean - rfr)) / sharpe_sq
        w_rfr = 1 - w.T @ np.ones(len(w))

    # tangency portfolio weights
    elif tangent:
        w = (inv_sigma @ (mean - rfr)) / (b - a * rfr)
        w_rfr = 0

    # weights without risk-free asset
    else:
        w = (c - ret * b) / delta * inv_sigma @ np.ones(n) + (ret * a - b) / delta * inv_sigma @ mean
        w_rfr = 0

    return w, w_rfr


def tangency_portfolio(df: pd.DataFrame, rf: float, long_only: bool):
    """
    maximizes sharpe ratio to find the tangency portfolio
    ---
    :param df: dataframe excluding the risk-free asset
    :param rf: risk-free rate
    :param long_only: boolean determining whether we desire the long only or unconstrained tangency portfolio
    :return: returns the weights of the tangency portfolio
    """
    mean = df.mean(axis=0)
    sigma = df.cov()

    # initial weights
    init_w = np.full(shape=len(mean), fill_value=1 / len(mean))
    # constraint
    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})  # weights must sum to 1

    if long_only:
        bounds = []
        for i in range(len(mean)):
            bounds.append((0, None))

        bounds = tuple(bounds)

        response = optimize.minimize(neg_sharpe_ratio, init_w,
                                     (sigma, mean, rf),
                                     constraints=cons,
                                     bounds=bounds,
                                     method="SLSQP")

    else:
        response = optimize.minimize(neg_sharpe_ratio, init_w,
                                     (sigma, mean, rf),
                                     constraints=cons,
                                     method="SLSQP")

    return response


def global_min_var_portfolio(mean: np.array, sigma: np.array, long_only: bool):
    """
    finds return of global minimum variance portfolio
    :param mean: mean returns of each asset in portfolio
    :param sigma: variance-covariance matrix of portfolio of assets
    :param long_only: boolean adding a no short sale constraint
    :return: return of GMVP
    """

    m = gp.Model()
    m.Params.LogToConsole = 0

    if long_only:
        w = m.addMVar(len(mean), lb=0, ub=1, name="weights")
    else:
        w = m.addMVar(len(mean), lb=-np.inf, ub=np.inf, name="weights")

    m.addConstr(w.sum() == 1, name="Budget_Constraint")

    m.setObjective(w @ sigma.to_numpy() @ w, gp.GRB.MINIMIZE)
    m.optimize()

    global_min_var_ret = float(m.X @ mean)

    return global_min_var_ret


def max_return_gurobi(mean: np.array, sigma: np.array, var: float, long_only: bool,
                      below_mvp: bool, ret_mvp: float):
    """
    minimizes variance of portfolio using sequential least squares programming and returns minimized std
    :param mean: mean returns of each asset in portfolio
    :param sigma: variance-covariance matrix of portfolio of assets
    :param var: the target variance for the portfolio against which we maximize return
    :param below_mvp: boolean determining whether we constrain the target return to be above or below the GMVP
    :param long_only: boolean adding a no short sale constraint
    :param ret_mvp: return of the minimum variance portfolio
    :return: standard deviation of portfolio
    """

    m = gp.Model()
    m.Params.LogToConsole = 0

    if long_only:
        w = m.addMVar(len(mean), lb=0, ub=1, name="weights")
        m.addConstr(w.sum() == 1, name="Budget_Constraint")

    else:
        w = m.addMVar(len(mean), lb=-np.inf, ub=np.inf, name="weights")
        m.addConstr(w.sum() == 1, name="Budget_Constraint")

    if below_mvp:
        m.addConstr(w @ mean.to_numpy() <= ret_mvp)

    m.addConstr(w @ sigma.to_numpy() @ w == var, name="Min_Return")

    m.setObjective(w @ mean.to_numpy(), gp.GRB.MAXIMIZE)
    m.optimize()

    return m


def eff_frontier(df: pd.DataFrame,
                 target_vars: np.array,
                 n: int,
                 risk_free_asset: bool,
                 long_only: bool,
                 tangent: bool, optimizer: str):
    """
    computes the return and standard deviation of each mean variance portfolio for a given expected return
    :param df: dataframe containing the returns of each industry (rows are dates, columns are industries)
    :param target_returns: list of expected returns we construct portfolios to target to build locus
    :param n: number of industries
    :param risk_free_asset: boolean determining whether we're building portfolio with or without risk-free asset
    :param long_only: boolean determining whether we're building long only portfolio or not
    :param tangent: boolean determining whether we're tangent portfolio or not
    :param optimizer: written 'scipy' or 'gurobi'. If gurobi, uses gurobi, else uses scipy
    :return:
    """
    # first and second moment
    cov = df.cov()
    mean = df.mean(axis=0)

    x = np.full(shape=len(target_vars), fill_value=None)  # std
    y = np.full(shape=len(target_vars), fill_value=None)  # target return
    w = np.full(shape=len(target_vars), fill_value=None)  # weights
    if optimizer == "scipy":
        # account for removal of risk-free asset in var-covar matrix
        if risk_free_asset:
            cov = df.iloc[:, 1:].cov()

        for j in range(len(target_vars)):
            ret = target_vars[j]
            response = minimize_variance(mean, cov, ret, n, risk_free_asset, long_only, tangent)
            x[j] = response.fun
            y[j] = mean @ response.x
            w[j] = response.x

    elif optimizer == "gurobi":

        min_var_ret = global_min_var_portfolio(mean, cov, long_only)

        for j in range(len(target_vars)):

            if target_vars[j] < 0:
                below_mvp = True
            else:
                below_mvp = False

            var = target_vars[j] ** 2  # since we're setting target volatilities, square to get target returns
            response = max_return_gurobi(mean, cov, var, long_only, below_mvp, min_var_ret)

            try:
                x[j] = np.sqrt(response.X @ cov @ response.X)
                y[j] = mean @ response.X
                w[j] = response.X
            except AttributeError:
                x[j] = None
                y[j] = None
                w[j] = [None] * n
            except gp._exception.GurobiError:
                w[j] = [None] * n

    else:
        ValueError("Use Gurobi or Scipy as optimizer input")

    return [x, y], w

def max_sharpe_ratio(df: pd.DataFrame, exp_return: float, short_constraint: bool) -> tuple:
    """ 
    Computes the maximum sharpe ratio for a given return

    Inputs
        df: dataframe of monthly returns, each colum is a different asset
        exp_return: expected return for the sharpe ratio
        short_constraint: boolean for short constraint 
    
    Outputs
        x: weights of the assets
        SR: max sharpe ratio for the specified return
        result: dataframe of optimized positions 
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
    
    b = m.addMVar(len(mean), vtype=gp.GRB.BINARY, name="b")

    if short_constraint:
        x = m.addMVar(len(mean), lb=-np.inf, ub=np.inf, name="x")
        x_plus = m.addMVar(len(mean), lb= 0, ub=np.inf, name="x_plus")
        x_minus = m.addMVar(len(mean), lb= 0, ub=np.inf, name="x_minus")

        # Budget constraint: all investments sum up to 1
        m.addConstr(x == x_plus - x_minus, name= "Position_Balance")
        m.addConstr(x_plus <= l*b, name= "Long_Indicator")
        m.addConstr(x_minus <= l*b, name= "Short_Indicator")
    else:
        x = m.addMVar(len(mean), lb=0, ub=np.inf, name="x")
        m.addConstr(x <= l*b, name= "Long_Indicator")

    # Budget constraint: all investments sum up to 1
    
    m.addConstr(x.sum() == 1, name="Budget_Constraint")
    m.addConstr(b.sum() <= K, name="Cardinality")
    m.addConstr(x.T @ mean.to_numpy() >= exp_return , name="Target_Return")

    # Minimize variance
    m.setObjective(x.T @ cov.to_numpy() @ x, gp.GRB.MINIMIZE)

    m.optimize()

    try:
        var = x.X @ cov.to_numpy() @ x.X
        rets = mean @ x.X
        SR = rets/np.sqrt(var)

        positions = pd.Series(name="Position", data= x.X, index= mean.index)
        index = positions[abs(positions) > 1e-5].index
        result = pd.DataFrame({'mean' : df[index].mean(),
                                'var' : df[index].var()})
    except:
        SR= None
        x = None
        result = None

    return SR, x, result
