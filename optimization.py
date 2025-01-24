import numpy as np
import pandas as pd
from scipy import optimize
from statsmodels.stats.correlation_tools import cov_nearest
import gurobipy as gp


def portfolio_std(w: np.array, sigma: np.array, risk_free_asset: bool):
    """
    computes variance of portfolio
    :param w: weights of each asset in portfolio (assume
    :param sigma: variance-covariance matrix of portfolio assets
    :param risk_free_asset: boolean determining whether there is a risk free asset
    :return: standard deviation of portfolio
    """

    if risk_free_asset:
        return np.sqrt(w[1:].T @ sigma @ w[1:])
    else:
        return np.sqrt(w.T @ sigma @ w)


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

    x = []  # std
    y = []  # target return
    w = []  # weights
    if optimizer == "scipy":
        # account for removal of risk-free asset in var-covar matrix
        if risk_free_asset:
            cov = df.iloc[:, 1:].cov()

        for ret in target_returns:
            response = minimize_variance(mean, cov, ret, n, risk_free_asset, long_only, tangent)
            x.append(response.fun)
            y.append(ret)
            w.append(response.x)

    elif optimizer == "gurobi":

        for ret in target_returns:
            response = minimize_variance_gurobi(mean, cov, ret, risk_free_asset, long_only, tangent)

            try:
                x.append(np.sqrt(response.ObjVal))
                y.append(ret)
                w.append(response.X)
            except AttributeError:
                x.append(None)
                y.append(None)
                w.append([None] * n)
            except gp._exception.GurobiError:
                w.append([None] * n)

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
    if risk_free_asset and tangent:
        ValueError("Set risk-free asset to True to compute tangency portfolio")

    # weights with risk-free asset
    elif risk_free_asset:
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
