from matplotlib import pyplot as plt
from adjustText import adjust_text


def mean_var_locus(std_portfolio, mean_portfolio, std_assets, mean_assets, title):
    fig, ax = plt.subplots()

    ax.plot(std_portfolio, mean_portfolio, linewidth=2)
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
