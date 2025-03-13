import numpy as np
import pandas as pd
from scipy.optimize import minimize

def compute_portfolio_weights(returns_df, rf_series, market_cap_df=None, window_length=None):
    """
    Compute portfolio weights for each month based on a rolling window.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with monthly returns for 10 industry portfolios (columns) and datetime index.
    rf_series : pd.Series
        Series with monthly risk-free rate (same index as returns_df).
    market_cap_df : pd.DataFrame, optional
        DataFrame with monthly market capitalizations (same index and columns as returns_df).
    window_length : int, default=60
        The number of months to use for the rolling window (default is 60 months = 5 years).
    
    Returns
    -------
    portfolios : dict
        Dictionary with keys corresponding to portfolio types and values as DataFrames of weights.
        The keys are:
          - 'max_sharpe_unconstrained': Unconstrained tangency (max Sharpe) portfolio.
          - 'max_sharpe_constrained': Tangency portfolio with no short sales (weights >= 0).
          - 'inv_variance': Weights inversely proportional to the asset’s variance.
          - 'inv_volatility': Weights inversely proportional to the asset’s volatility.
          - 'equal_weight': Equal weight portfolio.
          - 'market_cap': Weights proportional to market capitalization (if market_cap_df provided).
          - 'min_variance': Unconstrained minimum variance portfolio.
          
    Notes
    -----
    The function computes the weights using the past `window_length` months of excess returns 
    (i.e. returns minus risk-free rate). For each rebalancing date t, the weights computed 
    at time t are assumed to be applied for the following month.
    """
  
    port_names = ['max_sharpe_unconstrained', 'max_sharpe_constrained',
                  'inv_variance', 'inv_volatility', 'equal_weight',
                  'market_cap', 'min_variance']
    
    # Initialize dictionary to store results
    results = {name: [] for name in port_names if (name != 'market_cap' or market_cap_df is not None)}
    dates = []
    asset_names = returns_df.columns.tolist()
    n_assets = len(asset_names)
    
    for i in range(window_length - 1, len(returns_df) - 1):

        date = returns_df.index[i]
        dates.append(returns_df.index[i + 1])  # assign next month as the portfolio return date
        
        # Extract rolling window data
        window_returns = returns_df.iloc[i - window_length + 1 : i + 1]
        window_rf = rf_series.iloc[i - window_length + 1 : i + 1]
        
        # Compute excess returns
        window_excess = window_returns.subtract(window_rf, axis=0)
        
        # Compute mean excess returns and covariance matrix
        mu_excess = window_excess.mean().values
        Sigma = window_excess.cov().values
        
        # 1) Max Sharpe (unconstrained):
        def neg_sharpe_unconstrained(w):
            port_excess = np.dot(w, mu_excess)
            vol = np.sqrt(np.dot(w, Sigma.dot(w)))
            return -1 * port_excess / vol

        cons_uncon = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        init_guess_uncon = np.full(n_assets, 1 / n_assets)
        opt_uncon = minimize(neg_sharpe_unconstrained, init_guess_uncon, method='SLSQP', constraints=cons_uncon)
        w1 = opt_uncon.x if opt_uncon.success else np.full(n_assets, np.nan)
        
        # 2) Max Sharpe (constrained, no short sales)
        def neg_sharpe(w):
            port_excess = np.dot(w, mu_excess)
            vol = np.sqrt(np.dot(w, Sigma.dot(w)))
            return -1 * port_excess / vol
        
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_assets)]
        init_guess = np.full(n_assets, 1 / n_assets)
        opt = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        w2 = opt.x if opt.success else np.full(n_assets, np.nan)
        
        # 3) Inverse variance portfolio
        variances = np.diag(Sigma)
        inv_var = 1 / variances
        w3 = inv_var / np.sum(inv_var)
        
        # 4) Inverse volatility portfolio
        volatilities = np.sqrt(variances)
        inv_vol = 1 / volatilities
        w4 = inv_vol / np.sum(inv_vol)
        
        # 5) Equal weight portfolio
        w5 = np.full(n_assets, 1 / n_assets)
        
        # 6) Market cap weighted portfolio
        if market_cap_df is not None:
            market_cap = market_cap_df.loc[date].values
            w6 = market_cap / np.sum(market_cap)
        
        # 7) Minimum variance portfolio (numerical solution)
        def variance(w):
            return np.dot(w, Sigma.dot(w))

        cons_minvar = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        init_guess_minvar = np.full(n_assets, 1 / n_assets)
        opt_minvar = minimize(variance, init_guess_minvar, method='SLSQP', constraints=cons_minvar)
        w7 = opt_minvar.x if opt_minvar.success else np.full(n_assets, np.nan)
        
        # Append the computed weights
        results['max_sharpe_unconstrained'].append(w1)
        results['max_sharpe_constrained'].append(w2)
        results['inv_variance'].append(w3)
        results['inv_volatility'].append(w4)
        results['equal_weight'].append(w5)
        if market_cap_df is not None:
            results['market_cap'].append(w6)
        results['min_variance'].append(w7)
    
    # Convert lists to dataframes with proper index and column names
    portfolios = {}
    for key, weights_list in results.items():
        portfolios[key] = pd.DataFrame(weights_list, index=dates, columns=asset_names)
    
    return portfolios