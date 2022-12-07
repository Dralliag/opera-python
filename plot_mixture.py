import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_weight(ax, labels, colors, x):
    # Stack plot of weights associated to each expert
    ax.stackplot(
        range(len(x.weights)),
        np.stack(x.weights).T,
        edgecolor="white",
        colors=colors,
        labels=labels,
    )
    ax.set_title("Weights associated with the experts")
    ax.set(ylabel="Weights")
    ax.grid()


def boxplot_weight(ax, labels, colors, x):
    # Boxplot of weights associated to each expert
    idx = np.argsort(np.mean(x.weights, 0))[::-1]
    handles = ax.boxplot(
        np.stack(x.weights)[:, idx],
        showfliers=False,
        patch_artist=True,
        labels=labels[idx],
    )
    for box, c in zip(ax.patches, colors[idx]):
        box.set_facecolor(c)
    ax.set_title("Weights associated with the experts")
    ax.set(ylabel="Weights")
    ax.grid()
    return handles


def avg_loss(ax, alabels, unimix, colors, x):
    preds = np.column_stack((x.experts, x.predictions, unimix))
    loss = np.array([x.loss_type(x.targets, pred) for pred in preds.T])
    # epsilon = np.min(residuals.mean(1)) * 0.99
    sortedloss = np.sort(loss.mean(1))  # - epsilon
    idx = np.argsort(loss.mean(1))
    ax.bar(alabels[idx], sortedloss, color=colors[idx], alpha=1, label=alabels[idx])
    ax.set_title("Average loss suffered by the experts")
    ax.set(ylabel="Average Loss")
    ax.grid()


def cumul_res(ax, alabels, unimix, colors, x):
    preds = np.column_stack((x.experts, x.predictions, unimix))
    cumres = np.cumsum([x.targets - pred for pred in preds.T], 1).T
    for i in range(2, cumres.shape[1]):
        ax.plot(cumres[:, i], color=colors[i], label=alabels[i])
    ax.plot(cumres[:, 0], color=colors[0], label=alabels[0])
    ax.plot(cumres[:, 1], color=colors[1], label=alabels[1])
    ax.set_title("Cumulative Residuals")
    ax.set(ylabel="Cumulative Residuals")
    ax.grid()


def dyn_avg_loss(ax, alabels, unimix, colors, x):
    preds = np.column_stack((x.experts, x.predictions, unimix))
    cumloss = np.cumsum([x.loss_type(x.targets, pred) for pred in preds.T], 1).T
    for i in range(0, cumloss.shape[1]):
        ax.plot(cumloss[:, i], color=colors[i], label=alabels[i])
    # ax.plot(cumloss[:, 0], color=colors[0], label=alabels[0])
    # ax.plot(cumloss[:, 1], color=colors[1], label=alabels[1])
    ax.set_title("Dynamic average loss")
    ax.set(ylabel="Cumulative Loss")
    ax.grid()


def contrib(ax, labels, colors, x):
    # Stack plot of weights associated to each expert
    ax.stackplot(
        range(len(x.weights)),
        np.stack(x.weights).T * x.predictions,
        edgecolor="white",
        colors=colors,
        labels=labels,
    )
    ax.plot(
        range(len(x.predictions)),
        x.predictions,
        color="black",
        linestyle="dashed",
        label="Predictions",
    )
    ax.set_title("Contribution of each expert to the prediction")
    ax.set(ylabel="Contributions")
    ax.grid()


def plot_mixture(x, plot_type="all", colors=None):
    figsize = (10, 8)
    if len(np.array(x.experts).shape) > 2:
        print("Is plotted only the first dimension of the experts.")
        x.experts = np.array(x.experts).T[0, :, :]
        x.targets = np.array(x.targets)[:, 0]
        x.predictions = x.predictions[:, 0]
        x.weights = np.stack(x.weights)[:, 0, :]
    else:
        x.experts = np.array(x.experts)
        x.targets = np.array(x.targets)
    # K est le nombre d'experts
    K = x.experts.shape[1]
    unimix = np.sum(x.experts * np.ones_like(x.experts) / K, 1)

    if colors is None:
        colors = sns.color_palette(None, K + 2)
    colors = np.array(colors)

    if K <= 10:
        labels = np.array(x.experts_names)
        alabels = np.hstack((labels, x.model, "Uniform"))
    else:
        labels = np.array([str(i) for i in range(K)])
        alabels = np.hstack((labels, x.model[0], "U"))
    if plot_type == "all":
        fig, ax = plt.subplots(3, 2, figsize=figsize, dpi=100)
        # Stack plot of weights associated to each expert
        plot_weight(ax[0, 0], labels, colors, x)
        # Boxplot of weights associated to each expert
        boxplot_weight(ax[0, 1], labels, colors, x)
        # Barplot loss
        dyn_avg_loss(ax[1, 0], alabels, unimix, colors, x)
        # Cumulative residuals
        cumul_res(ax[1, 1], alabels, unimix, colors, x)

        # Cumulative loss
        avg_loss(ax[2, 0], alabels, unimix, colors, x)

        # Cumulative loss
        contrib(ax[2, 1], labels, colors, x)

        handles, labels = ax[1, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=10, borderaxespad=1.0)
        fig.suptitle(" ", fontsize=24)
        fig.tight_layout()
    elif plot_type == "boxplot_weight":
        fig, ax = plt.subplots(dpi=100)
        # Boxplot of weights associated to each expert
        handles = boxplot_weight(ax, labels, colors, x)
        fig.legend(
            handles["boxes"], labels, loc="upper center", ncol=K + 2, borderaxespad=1.0
        )
        fig.suptitle(" ", fontsize=16)
        fig.tight_layout()
    elif plot_type == "plot_weight":
        fig, ax = plt.subplots(dpi=100)
        # Stack plot of weights associated to each expert
        plot_weight(ax, labels, colors, x)
        fig.legend(loc="upper center", ncol=K + 2, borderaxespad=1.0)
        fig.suptitle(" ", fontsize=16)
        fig.tight_layout()
    elif plot_type == "contrib":
        fig, ax = plt.subplots(dpi=100)
        # Stack plot of weights associated to each expert
        contrib(ax, labels, colors, x)
        fig.legend(loc="upper center", ncol=K + 2, borderaxespad=1.0)
        fig.suptitle(" ", fontsize=16)
        fig.tight_layout()
    elif plot_type == "dyn_avg_loss":
        fig, ax = plt.subplots(dpi=100)
        # Barplot loss
        dyn_avg_loss(ax, alabels, unimix, colors, x)
        fig.legend(loc="upper center", ncol=K + 2, borderaxespad=1.0)
        fig.suptitle(" ", fontsize=16)
        fig.tight_layout()
    elif plot_type == "cumul_res":
        fig, ax = plt.subplots(dpi=100)
        # Cumulative residuals
        cumul_res(ax, alabels, unimix, colors, x)
        fig.legend(loc="upper center", ncol=K + 2, borderaxespad=1.0)
        fig.suptitle(" ", fontsize=16)
        fig.tight_layout()
    elif plot_type == "avg_loss":
        fig, ax = plt.subplots(dpi=100)
        # Cumulative loss
        avg_loss(ax, alabels, unimix, colors, x)
        fig.legend(loc="upper center", ncol=K + 2, borderaxespad=1.0)
        fig.suptitle(" ", fontsize=16)
        fig.tight_layout()
    else:
        raise (NotImplementedError(f"{plot_type} plot not implemented yet."))
    # TODO max experts, prendre que les meilleurs experts et aggréger les pires
    plt.show()
