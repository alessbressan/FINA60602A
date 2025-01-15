import numpy as np
from scipy import optimize


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


def minimize_variance(mean: np.array, sigma: np.array, ret: float, n: int, risk_free_asset: bool, long_only: bool, tangent: bool):
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
                                     method="SLSQP",
                                     options= {'maxiter' : 500})

    else:
        response = optimize.minimize(portfolio_std, init_w,
                                     (sigma, risk_free_asset),
                                     constraints=cons,
                                     method="SLSQP",
                                     options= {'maxiter' : 500})

    if not response.success:
        print(f'\nResponse: {response}')
        print(f'\nmean:\n {mean}')
        print(f'\nreturns:\n {ret}')

        raise RuntimeError(f"Optimization failed: {response.message}")
    
    return response

