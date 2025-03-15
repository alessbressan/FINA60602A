import numpy as np
import pandas as pd
from scipy import optimize
from statsmodels.stats.correlation_tools import cov_nearest
import gurobipy as gp

def minimize_variance_gurobi(mean: np.array, sigma: np.array, ret: float, risk_free_asset: bool= False, long_only: bool= False,
                                  tangent: bool= False):
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

    b = m.addMVar(len(mean), vtype=gp.GRB.BINARY, name="b")

    if long_only:
        w = m.addMVar(len(mean), lb=0, ub=np.inf, name="weights")
        m.addConstr(w.sum() == 1, name="Budget_Constraint")
        m.addConstr(w <= b, name= "Indicator")

        if (tangent and risk_free_asset):
            m.addConstr(w[0] == 0, name="Tangency_Constraint")

        elif tangent and not risk_free_asset:
            ValueError("Set risk-free asset to True to compute tangency portfolio")

    else:
        limit = 100 # limit
        w = m.addMVar(len(mean), lb=-np.inf, ub=np.inf, name="weights")
        w_plus = m.addMVar(len(mean), lb= 0, ub= np.inf, name="weights_pos")
        w_minus = m.addMVar(len(mean), lb= 0, ub= np.inf, name="weights_neg")
        
        m.addConstr(w.sum() == 1, name="Budget_Constraint")
        m.addConstr(w == w_plus - w_minus, name= "Position_Balance")
        m.addConstr(w_plus <= limit*b, name= "Long_Indicator")
        m.addConstr(w_minus <= limit*b, name= "Short_Indicator")

        if tangent and not risk_free_asset:
            ValueError("Set risk-free asset to True to compute tangency portfolio")

    # the return constraint is useless when considering the tangent portfolio
    # this is because the tangent portfolio already requires specific return by forcing risk-free weight to zero
    if not tangent:
        m.addConstr(mean.to_numpy() @ w == ret, name="Min_Return")


    m.addConstr(b.sum() <= 3, name= "Cardinality")

    m.setObjective(w @ sigma.to_numpy() @ w, gp.GRB.MINIMIZE)
    m.optimize()
    try:
        return m.ObjVal, w.X
    except Exception as e:
        # print(f'Error: {e}')
        return None, None

def mean_var_portfolio(df: pd.DataFrame,
                       target_returns: np.array,
                       n: int,
                       risk_free_asset: bool= False,
                       long_only: bool= False,
                       tangent: bool= False):
    """
    computes the return and standard deviation of each mean variance portfolio for a given expected return
    :param df: dataframe containing the returns of each industry (rows are dates, columns are industries)
    :param target_returns: list of expected returns we construct portfolios to target to build locus
    :param n: number of industries
    :param risk_free_asset: boolean determining whether we're building portfolio with or without risk-free asset
    :param long_only: boolean determining whether we're building long only portfolio or not
    :param tangent: boolean determining whether we're tangent portfolio or not
    :param optimizer: written 'scipy' or 'gurobi'. If gurobi, uses gurobi, else uses scipy
    :param cardinality: boolean to add cardinality constraint
    :return:
    """
    # first and second moment
    cov = df.cov()
    mean = df.mean(axis=0)

    x = np.full(shape=len(target_returns), fill_value=None)  # std
    y = np.full(shape=len(target_returns), fill_value=None)  # target return
    w = np.full(shape=len(target_returns), fill_value=None)  # weights
    for j in range(len(target_returns)):
        ret = target_returns[j]

        obj_val, weights = minimize_variance_gurobi(mean, cov, ret, risk_free_asset, long_only, tangent)

        try:
            x[j] = np.sqrt(obj_val)
            y[j] = mean @ weights
            w[j] = weights
        except AttributeError as e:
            print(f'Error message: {e}')
            x[j] = None
            y[j] = None
            w[j] = [None] * n
        
        except gp._exception.GurobiError:
            w[j] = [None] * n

    return [x, y], w

def max_sharpe_ratio(df: pd.DataFrame, exp_return: float, K= 3, long_only: bool= False) -> tuple:
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

    l = 100 #max position size (500%)

    # Create an empty optimization model
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads=8

    # Add variables: x[i] denotes the proportion of capital invested in stock i
    
    b = m.addMVar(len(mean), vtype=gp.GRB.BINARY, name="b")

    if not long_only:
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
        return None, None, None

    return SR, x.X, result

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