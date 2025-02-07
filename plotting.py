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


def tangency_plot(std_portfolio_rf, mean_portfolio_rf, std_portfolio_no_rf, mean_portfolio_no_rf, tangency_mean,
                  tangency_std, std_assets, mean_assets, title):
    fig, ax = plt.subplots()

    ax.plot(std_portfolio_no_rf, mean_portfolio_no_rf, linewidth=1, color="black")
    ax.plot(std_portfolio_rf, mean_portfolio_rf, linewidth=1, color="black")
    ax.scatter(tangency_std, tangency_mean, c="red")
    ax.scatter(std_assets, mean_assets, c=[*range(0, len(mean_assets.index))], cmap="viridis")

    texts = []
    for i in range(len(mean_assets.index)):
        texts.append(plt.text(std_assets[i], mean_assets[i], mean_assets.index[i], fontsize=10))
        # ax.scatter(std[i], mean[i], color=i)

    texts.append(plt.text(tangency_std, tangency_mean, "Tangency Portfolio", fontsize=10))

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    ax.set(xlabel='Standard Deviation (%)', ylabel='Expected Return (%)',
           title=title)
    ax.grid()
    plt.show()


def confidence_bands(original_data: pd.DataFrame,
                     bootstrapped_data: np.array,
                     target_returns: np.array,
                     n: int,
                     conf_lvl: float,
                     title: str,
                     risk_free_asset: bool,
                     long_only: bool):
    """
    plots the original mean variance locus and 95% confidence bands to demonstrate parameter instability

    :param original_data: dataframe containing the returns of the original data
    :param bootstrapped_data: nb_bootstraps x len(target_returns) x 2 (last dimension contains [std, return])
    :param target_returns: list of returns targeted in portfolio
    :param n: number of assets in the portfolio
    :conf_lvl: displays the % confidence level of the confidence interval
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
        if all(np.isnan(bootstrapped_data[:, j, 0])):
            u_quantile.append(None)
            l_quantile.append(None)
        else:
            u_quantile.append(np.quantile(bootstrapped_data[:, j, 0][~np.isnan(bootstrapped_data[:, j, 0])], conf_lvl/2))
            l_quantile.append(np.quantile(bootstrapped_data[:, j, 0][~np.isnan(bootstrapped_data[:, j, 0])], 1-conf_lvl/2))

    # find true mean variance locus based on our current data
    response, weights = optimization.mean_var_portfolio(original_data, target_returns, n, risk_free_asset, long_only,
                                                        False, "scipy")

    # plot
    fig, ax = plt.subplots()

    ax.plot(response[0], response[1], linewidth=2, color="black", label="Original Locus")
    ax.plot(l_quantile, target_returns, linewidth=1.5, color="pink", linestyle="--", label="Lower 95% CI")
    ax.plot(u_quantile, target_returns, linewidth=1.5, color="gray", linestyle="--", label="Upper 95% CI")

    # scatterplot of assets
    ax.scatter(std_assets, mean_assets, c=[*range(0, len(mean_assets.index))], cmap="viridis")

    texts = []
    for i in range(len(mean_assets.index)):
        texts.append(plt.text(std_assets[i], mean_assets[i], mean_assets.index[i], fontsize=10))

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    ax.set(xlabel='Standard Deviation (%)', ylabel='Expected Return (%)',
           title=title)
    ax.grid()
    ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
    plt.show()


def confidence_bands_max_ret(original_data: pd.DataFrame,
                             bootstrapped_data: np.array,
                             target_vols: np.array,
                             target_rets: np.array,
                             n: int,
                             title: str,
                             risk_free_asset: bool,
                             long_only: bool,
                             upper_portion: bool):
    """
    plots the original mean variance locus and 90% confidence bands to demonstrate parameter instability

    :param original_data: dataframe containing the returns of the original data
    :param bootstrapped_data: nb_bootstraps x len(target_returns) x 2 (last dimension contains [std, return])
    :param target_vols: list of variances targeted in portfolio
    :param target_rets: array of target returns
    :param n: number of assets in the portfolio
    :param title: title displayed
    :param risk_free_asset: boolean determining whether we included a risk-free asset
    :param long_only: boolean determining whether we had a short constraint
    :param upper_portion: if true, returns only the upper portion of the mean variance locus. Else, returns lower portion
    :return: plot of mean-variance locus with confidence bounds
    """
    plotting_target_vars = []
    u_quantile = []
    l_quantile = []

    mean_assets = original_data.mean(axis=0)
    std_assets = np.sqrt(np.diag(original_data.cov()))


    # find true mean variance locus based on our current data
    response, weights = optimization.mean_var_portfolio(original_data, target_rets, n, risk_free_asset, long_only,
                                                        False, "scipy")

    response = np.array(response)
    global_min_portfolio_ret = response[1][np.argmin(response[0])]
    # plot
    fig, ax = plt.subplots()

    if upper_portion:
        upper_std = response[0][response[1] > global_min_portfolio_ret]
        upper_rets = response[1][response[1] > global_min_portfolio_ret]
        ax.plot(upper_std, upper_rets, linewidth=2, color="black")
    else:
        lower_std = response[0][response[1] < global_min_portfolio_ret]
        lower_rets = response[1][response[1] < global_min_portfolio_ret]
        ax.plot(lower_std, lower_rets, linewidth=2, color="black")

    for j in range(bootstrapped_data.shape[1]):

        if upper_portion:
            if (target_vols[j] > min(response[0])) and (target_vols[j] < max(response[0])):
                try:
                    u_quantile.append(np.quantile(bootstrapped_data[:, j, 1][~np.isnan(bootstrapped_data[:, j, 1])], 0.1))
                    l_quantile.append(np.quantile(bootstrapped_data[:, j, 1][~np.isnan(bootstrapped_data[:, j, 1])], 0.9))
                    plotting_target_vars.append(abs(target_vols[j]))
                except IndexError:
                    pass
        else:
            if (-1 * target_vols[j] > min(response[0])) and (-1 * target_vols[j] < max(response[0])):
                try:
                    u_quantile.append(np.quantile(bootstrapped_data[:, j, 1][~np.isnan(bootstrapped_data[:, j, 1])], 0.1))
                    l_quantile.append(np.quantile(bootstrapped_data[:, j, 1][~np.isnan(bootstrapped_data[:, j, 1])], 0.9))
                    plotting_target_vars.append(abs(target_vols[j]))
                except IndexError:
                    pass


    ax.plot(plotting_target_vars, l_quantile, linewidth=1.5, color="pink", linestyle="--")
    ax.plot(plotting_target_vars, u_quantile, linewidth=1.5, color="gray", linestyle="--")

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
