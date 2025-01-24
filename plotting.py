import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from adjustText import adjust_text

import optimization


def mean_var_locus(std_portfolio, mean_portfolio, std_assets, mean_assets, title):
    fig, ax = plt.subplots()

    ax.plot(std_portfolio, mean_portfolio, linewidth=2, color="black")
    ax.scatter(std_assets, mean_assets, c=[*range(0, len(mean_assets.index))], cmap="viridis")

    texts = []
    for i in range(len(mean_assets.index)):
        texts.append(plt.text(std_assets[i], mean_assets[i], mean_assets.index[i], fontsize=10))
        # ax.scatter(std[i], mean[i], color=i)

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    ax.set(xlabel='Standard Deviation (%)', ylabel='Expected Return (%)',
           title=title)
    ax.grid()
    plt.show()


def confidence_bands(original_data: pd.DataFrame,
                     bootstrapped_data: np.array,
                     target_returns: np.array,
                     n: int,
                     title: str,
                     risk_free_asset: bool,
                     long_only: bool):
    """
    plots the original mean variance locus and 95% confidence bands to demonstrate parameter instability

    :param original_data: dataframe containing the returns of the original data
    :param bootstrapped_data: nb_bootstraps x len(target_returns) x 2 (last dimension contains [std, return])
    :param target_returns: list of returns targeted in portfolio
    :param n: number of assets in the portfolio
    :param title: title displayed
    :param risk_free_asset: boolean determining whether we included a risk-free asset
    :param long_only: boolean determining whether we had a short constraint
    :return: plot of mean-variance locus with confidence bounds
    """
    u_quantile = []
    l_quantile = []

    mean_assets = original_data.mean(axis=0)
    std_assets = np.sqrt(np.diag(original_data.cov()))

    for j in range(bootstrapped_data.shape[1]):
        u_quantile.append(np.quantile(bootstrapped_data[:, j, 0][~np.isnan(bootstrapped_data[:, j, 0])], 0.95))
        l_quantile.append(np.quantile(bootstrapped_data[:, j, 0][~np.isnan(bootstrapped_data[:, j, 0])], 0.05))

    # find true mean variance locus based on our current data
    response, weights = optimization.mean_var_portfolio(original_data, target_returns, n, risk_free_asset, long_only, False, "scipy")

    # plot
    fig, ax = plt.subplots()

    ax.plot(response[0], response[1], linewidth=2, color="black")
    ax.plot(l_quantile, target_returns, linewidth=1.5, color="gray", linestyle="--")
    ax.plot(u_quantile, target_returns, linewidth=1.5, color="gray", linestyle="--")

    # scatterplot of assets
    ax.scatter(std_assets, mean_assets, c=[*range(0, len(mean_assets.index))], cmap="viridis")

    texts = []
    for i in range(len(mean_assets.index)):
        texts.append(plt.text(std_assets[i], mean_assets[i], mean_assets.index[i], fontsize=10))

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    ax.set(xlabel='Standard Deviation (%)', ylabel='Expected Return (%)',
           title=title)
    ax.grid()
    plt.show()


def weights_boxplot(original_data: pd.DataFrame, bootstrapped_weights):

    pass