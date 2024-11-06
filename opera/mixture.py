"""
opera - Online Python by Expert Aggregation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Losses
def mape(x, y):
    return np.abs(x - y) / y


def gradient_mape(x, y):
    return 1 / y * np.sign(x - y)


def mae(x, y):
    return np.abs(x - y)


def gradient_mae(x, y):
    return np.sign(x - y)


def mse(x, y):
    return np.square(x - y)


def gradient_mse(x, y):
    return 2 * (x - y)


def msle(x, y):
    return np.square(np.log(y + 1) - np.log(x + 1))


def gradient_msle(x, y):
    return 2 * (np.log(y + 1) - np.log(x + 1)) * (-1 / (x + 1))


def mspe(x, y):
    return np.square(y - x) / np.square(y)


def gradient_mspe(x, y):
    return -2 * x + 2 * y


def normalize(x):
    return x / np.sum(x, axis=-1)


def idx_worst(arr, k):
    result = np.argpartition(arr, arr.shape[0] - k)
    return result[: arr.shape[0] - k]


def idx_best(arr, k):
    result = np.argpartition(arr, arr.shape[0] - k)
    return result[arr.shape[0] - k :]


def plot_weight(
    ax,
    colors,
    mixture,
    max_experts,
    title=None,
    ylabel=None,
    index_start=None,
    index_stop=None,
):

    # Stack plot of weights associated to each expert
    if title is None:
        title = "Weights associated with the experts"
    if ylabel is None:
        ylabel = "Weights"
    printable_weights = mixture.weights[index_start:index_stop].copy()
    # print(printable_weights)
    labels = np.array(mixture.experts_names)
    mean_weights = np.mean(printable_weights, axis=0)
    id_worst = idx_worst(mean_weights, max_experts)
    id_best = idx_best(mean_weights, max_experts)
    weights = printable_weights[:, id_best]
    colors = colors[id_best]
    labels = labels[id_best]
    if mixture.w.shape[0] > max_experts:
        avg_weights = np.sum(printable_weights[:, id_worst], axis=1, keepdims=True)
        weights = np.hstack([avg_weights, weights])
        colors = np.vstack([[0.6, 0.6, 0.6], colors])
        labels = np.hstack([["others"], labels])
    ax.stackplot(
        range(len(weights)),
        np.stack(weights).T,
        edgecolor="white",
        colors=colors,
        labels=labels,
    )
    ax.set_title(title)
    ax.set(ylabel=ylabel)
    ax.grid()


def boxplot_weight(
    ax,
    colors,
    mixture,
    max_experts,
    title=None,
    ylabel=None,
    index_start=None,
    index_stop=None,
):
    # Boxplot of weights associated to each expert

    if title is None:
        title = "Weights associated with the experts"
    if ylabel is None:
        ylabel = "Weights"
    labels = np.array(mixture.experts_names)
    printable_weights = mixture.weights[index_start:index_stop].copy()

    mean_weights = np.mean(printable_weights, axis=0)

    id_worst = idx_worst(mean_weights, max_experts)
    id_best = idx_best(mean_weights, max_experts)

    weights = printable_weights[:, id_best]
    colors = colors[id_best]
    labels = labels[id_best]
    if id_worst.shape[0] > 0:
        id_worst_worst = id_worst[np.argmin(mean_weights[id_worst])]
        worst_worst_weights = printable_weights[:, id_worst_worst]
        worst_worst_weights = worst_worst_weights.reshape(
            worst_worst_weights.shape[0], 1
        )
        weights = np.hstack([weights, worst_worst_weights])
        colors = np.vstack([colors, [0.5, 0.5, 0.5]])
        labels = np.hstack([labels, ["worst others"]])
    if id_worst.shape[0] > 1:
        id_best_worst = id_worst[np.argmax(mean_weights[id_worst])]
        best_worst_weights = printable_weights[:, id_best_worst]
        best_worst_weights = best_worst_weights.reshape(best_worst_weights.shape[0], 1)
        weights = np.hstack([weights, best_worst_weights])
        colors = np.vstack([colors, [0.7, 0.7, 0.7]])
        labels = np.hstack([labels, ["best others"]])

    idx = np.argsort(np.mean(weights, 0))[::-1]
    handles = ax.boxplot(
        np.stack(weights)[:, idx],
        showfliers=False,
        patch_artist=True,
        labels=labels[idx],
    )
    for box, c in zip(ax.patches, colors[idx]):
        box.set_facecolor(c)
    ax.set_title(title)
    ax.set_xticklabels(labels[idx], rotation=90)
    ax.set(ylabel=ylabel)
    ax.grid()

    return handles


def avg_loss(
    ax,
    colors,
    mixture,
    max_experts,
    title=None,
    ylabel=None,
    index_start=None,
    index_stop=None,
):

    if title is None:
        title = "Average loss suffered by the experts"
    if ylabel is None:
        ylabel = "Average Loss"
    labels = np.array(mixture.experts_names)
    K = mixture.experts.shape[1]
    printable_weights = mixture.weights[index_start:index_stop].copy()

    mean_weights = np.mean(printable_weights, axis=0)
    id_worst = idx_worst(mean_weights, max_experts)
    id_best = idx_best(mean_weights, max_experts)
    printable_experts = mixture.experts[index_start:index_stop].copy()
    experts = printable_experts[:, id_best]
    predictions = mixture.predictions[index_start:index_stop].copy()
    targets = mixture.targets[index_start:index_stop].copy()
    unimix = np.sum(printable_experts * np.ones_like(printable_experts) / K, 1)
    colors = colors[id_best]
    labels = labels[id_best]

    if id_worst.shape[0] > 0:
        id_worst_worst = id_worst[np.argmin(mean_weights[id_worst])]
        worst_worst_experts = printable_experts[:, id_worst_worst]
        worst_worst_experts = worst_worst_experts.reshape(
            worst_worst_experts.shape[0], 1
        )
        experts = np.hstack([experts, worst_worst_experts])
        colors = np.vstack([colors, [0.5, 0.5, 0.5]])
        labels = np.hstack([labels, ["worst others"]])
    if id_worst.shape[0] > 1:
        id_best_worst = id_worst[np.argmax(mean_weights[id_worst])]
        best_worst_experts = printable_experts[:, id_best_worst]
        best_worst_experts = best_worst_experts.reshape(best_worst_experts.shape[0], 1)
        experts = np.hstack([experts, best_worst_experts])
        colors = np.vstack([colors, [0.7, 0.7, 0.7]])
        labels = np.hstack([labels, ["best others"]])
    alabels = np.hstack((labels, mixture.model, "Uniform"))
    colors = np.vstack([colors, [0, 0, 0], [0.3, 0.3, 0.3]])
    preds = np.column_stack((experts, predictions, unimix))
    loss = np.array([mixture.loss_function(targets, pred) for pred in preds.T])
    sortedloss = np.sort(loss.mean(1))  # - epsilon
    idx = np.argsort(loss.mean(1))
    ax.bar(alabels[idx], sortedloss, color=colors[idx], alpha=1, label=alabels[idx])
    ax.set_title(title)
    ax.set_xticklabels(alabels[idx], rotation=90)
    ax.set(ylabel=ylabel)
    ax.grid()


def cumul_res(
    ax,
    colors,
    mixture,
    max_experts,
    title=None,
    ylabel=None,
    index_start=None,
    index_stop=None,
):

    if title is None:
        title = "Cumulative Residuals"
    if ylabel is None:
        ylabel = "Cumulative Residuals"
    labels = np.array(mixture.experts_names)
    K = mixture.experts.shape[1]
    printable_weights = mixture.weights[index_start:index_stop].copy()
    printable_awakes = mixture.awakes[index_start:index_stop].copy()
    printable_experts = mixture.experts[index_start:index_stop].copy()
    predictions = mixture.predictions[index_start:index_stop].copy()
    targets = mixture.targets[index_start:index_stop].copy()
    unimix = np.sum(printable_experts * np.ones_like(printable_experts) / K, 1)
    p_experts = printable_experts * printable_awakes + predictions.reshape(
        predictions.shape[0], 1
    ) * (1 - printable_awakes)
    mean_weights = np.mean(printable_weights, axis=0)
    id_worst = idx_worst(mean_weights, max_experts)
    id_best = idx_best(mean_weights, max_experts)
    pred_experts = p_experts[:, id_best]
    colors = colors[id_best]
    labels = labels[id_best]

    if id_worst.shape[0] > 0:
        id_worst_worst = id_worst[np.argmin(mean_weights[id_worst])]
        worst_worst_experts = p_experts[:, id_worst_worst]
        worst_worst_experts = worst_worst_experts.reshape(
            worst_worst_experts.shape[0], 1
        )
        pred_experts = np.hstack([pred_experts, worst_worst_experts])
        colors = np.vstack([colors, [0.5, 0.5, 0.5]])
        labels = np.hstack([labels, ["worst others"]])
    if id_worst.shape[0] > 1:
        id_best_worst = id_worst[np.argmax(mean_weights[id_worst])]
        best_worst_experts = p_experts[:, id_best_worst]
        best_worst_experts = best_worst_experts.reshape(best_worst_experts.shape[0], 1)
        pred_experts = np.hstack([pred_experts, best_worst_experts])
        colors = np.vstack([colors, [0.7, 0.7, 0.7]])
        labels = np.hstack([labels, ["best others"]])
    alabels = np.hstack((labels, mixture.model, "Uniform"))
    colors = np.vstack([colors, [0, 0, 0], [0.3, 0.3, 0.3]])
    preds = np.column_stack((pred_experts, predictions, unimix))
    cumres = np.cumsum([targets - pred for pred in preds.T], 1).T
    for i in range(2, cumres.shape[1]):
        ax.plot(cumres[:, i], color=colors[i], label=alabels[i])
    ax.plot(cumres[:, 0], color=colors[0], label=alabels[0])
    ax.plot(cumres[:, 1], color=colors[1], label=alabels[1])
    ax.set_title(title)
    ax.set(ylabel=ylabel)
    ax.grid()


def dyn_avg_loss(
    ax,
    colors,
    mixture,
    max_experts,
    title=None,
    ylabel=None,
    index_start=None,
    index_stop=None,
):

    if title is None:
        title = "Dynamic average loss"
    if ylabel is None:
        ylabel = "Average Loss"
    labels = np.array(mixture.experts_names)
    K = mixture.experts.shape[1]
    printable_weights = mixture.weights[index_start:index_stop].copy()
    printable_awakes = mixture.awakes[index_start:index_stop].copy()
    printable_experts = mixture.experts[index_start:index_stop].copy()
    unimix = np.sum(printable_experts * np.ones_like(printable_experts) / K, 1)
    predictions = mixture.predictions[index_start:index_stop].copy()
    targets = mixture.targets[index_start:index_stop].copy()

    p_experts = printable_experts * printable_awakes + predictions.reshape(
        predictions.shape[0], 1
    ) * (1 - printable_awakes)
    mean_weights = np.mean(printable_weights, axis=0)
    id_worst = idx_worst(mean_weights, max_experts)
    id_best = idx_best(mean_weights, max_experts)
    pred_experts = p_experts[:, id_best]
    colors = colors[id_best]
    labels = labels[id_best]

    if id_worst.shape[0] > 0:
        id_worst_worst = id_worst[np.argmin(mean_weights[id_worst])]
        worst_worst_experts = p_experts[:, id_worst_worst]
        worst_worst_experts = worst_worst_experts.reshape(
            worst_worst_experts.shape[0], 1
        )
        pred_experts = np.hstack([pred_experts, worst_worst_experts])
        colors = np.vstack([colors, [0.5, 0.5, 0.5]])
        labels = np.hstack([labels, ["worst others"]])
    if id_worst.shape[0] > 1:
        id_best_worst = id_worst[np.argmax(mean_weights[id_worst])]
        best_worst_experts = p_experts[:, id_best_worst]
        best_worst_experts = best_worst_experts.reshape(best_worst_experts.shape[0], 1)
        pred_experts = np.hstack([pred_experts, best_worst_experts])
        colors = np.vstack([colors, [0.7, 0.7, 0.7]])
        labels = np.hstack([labels, ["best others"]])
    alabels = np.hstack((labels, mixture.model, "Uniform"))
    colors = np.vstack([colors, [0, 0, 0], [0.3, 0.3, 0.3]])
    preds = np.column_stack((pred_experts, predictions, unimix))
    cumloss = np.cumsum([mixture.loss_function(targets, pred) for pred in preds.T], 1).T
    div = np.arange(1, preds.shape[0] + 1)
    div = np.repeat(div, preds.shape[1]).reshape(preds.shape)
    cumloss = cumloss / div
    for i in range(0, cumloss.shape[1]):
        ax.plot(cumloss[:, i], color=colors[i], label=alabels[i])

    ax.set_title(title)
    ax.set(ylabel=ylabel)

    ax.grid()


def contrib(
    ax,
    colors,
    mixture,
    max_experts,
    title=None,
    ylabel=None,
    index_start=None,
    index_stop=None,
):

    if title is None:
        title = "Contribution of each expert to the prediction"
    if ylabel is None:
        ylabel = "Contributions"

    # Stack plot of weights associated to each expert
    printable_weights = mixture.weights[index_start:index_stop].copy()
    printable_predictions = mixture.predictions[index_start:index_stop].copy()
    mean_weights = np.mean(printable_weights, axis=0)
    id_worst = idx_worst(mean_weights, max_experts)
    id_best = idx_best(mean_weights, max_experts)
    labels = np.array(mixture.experts_names)
    weights = printable_weights[:, id_best]
    colors = colors[id_best]
    labels = labels[id_best]
    if mixture.w.shape[0] > max_experts:
        avg_weights = np.sum(printable_weights[:, id_worst], axis=1, keepdims=True)
        weights = np.hstack([avg_weights, weights])
        colors = np.vstack([[0.6, 0.6, 0.6], colors])
        labels = np.hstack([["others"], labels])

    ax.stackplot(
        range(len(weights)),
        np.stack(weights).T * printable_predictions,
        edgecolor="white",
        colors=colors,
        labels=labels,
    )
    ax.plot(
        range(len(printable_predictions)),
        printable_predictions,
        color="black",
        linestyle="dashed",
        label="Predictions",
    )
    ax.set_title(title)
    ax.set(ylabel=ylabel)
    ax.grid()


class Mixture:
    """
    Abstract class for the mixture model, allowing to compute aggregation rules.

    Consider a sequence of real bounded observations (y[1],...,y[T]) to be predicted step by step.
    A finite set of methods (k =1,...,K) (henceforth referred to as experts) that provide you before
    each time step (t=1,...,T) predictions (x[k,t]) of the next observation y[t]).
    The prediction (^y[t]) can be formed by using only the knowledge of the past observations
    (y[1],...,y[t-1]) and past and current expert forecasts (x[k,1],...,x[kplot_type,t]) for (k=1,...,K).
    The package opera implements several algorithms of the onl_typeine learning literature that form
    predictions (^y[t]) by combining the expert forecasts according to their past performance.
    That is, [^y[t] = sum {k=1}^K p[k,t] x[k,t] ] These algorithms come with finite time worst-case
    guarantees. The monograph of [Cesa-Bianchi and Lugisi (2006)]
    (http://www.ii.uni.wroc.pl/~lukstafi/pmwiki/uploads/AGT/Prediction_Learning_and_Games.pdf)
    gives a complete introduction to the setting of prediction of arbitrary sequences with
    the help of expert advice.

    Args:
        y (numpy.array or pandas.DataFrame): array of targets
        experts (numpy.array or pandas.DataFrame): array of experts
        awake (numpy.array or pandas.DataFrame, optional): A matrix specifying the activation coefficients
            of the experts. Defaults to None.
        model (str, optional): string specifying the aggregation rule to use. Currently available aggregation
            rules are: BOA, MLpol, MLprod, FTRL. Defaults to "BOA".
        coefficients (array or str, optional): array of weight to be used for the aggregation rule. Defaults to "uniform".
        loss_type (function or string, optional): a custom function to evaluate the performances of the aggregation
        rule or a string specifying one of the available functions
            -Mean Absolute Percentage Error "mape",
            -Mean Absolute Error "mae",
            -Mean Squared Error "mse",
            -Mean Squared Logarithmic Error "msle",
            -Mean squared prediction Error "mspe".
            Defaults to mse.
        loss_gradient (function or bool, optional): the derivative of the custom loss function or a Boolean specifying
            whether the loss is used with gradient or no.. Defaults to True.
        parameters (dict): dict of optional parameters for FTRL algorithm, default to None, available parameters :
            - "fun_reg": objective function
            - "fun_reg_grad": gradient of the objective function
            - "constraints": liste of constraints to pass to the optimizer
            - "tol": tolerance for termination
            - "options": a dictionary of solver options
            For more informations on how to use the parameters, give a look to Example 4 below

    Attributes
    ----------
    predictions : history of predictions
    weights : history of weights
    awakes : history of awakes
    experts : history of experts
    targets : history of targets

    Methods
    -------

    updates(new_experts, new_y, awake): updates the model sequentially with new experts and new targets
    predict(new_experts, awake): Performs sequential predictions and updates of a mixture object based on new observations
        and the last coefficients
    plot_mixture(plot_type, colors) : provides different diagnostic plots for an aggregation procedure.

    Examples
    --------
    # Example 1
    import pandas as pd
    import numpy as np
    from opera import Mixture

    targets = pd.read_csv("data/targets.csv")["x"]
    experts = pd.read_csv("data/experts.csv")
    awake = np.tile(np.array([1, 0, 1]), (experts.shape[0],)).reshape(experts.shape)

    mod_1 = Mixture(
        y=targets.iloc[0:100],
        experts=experts.iloc[0:100],
        awake=awake[0:100],
        model="BOA",
        loss_type="mse",
        loss_gradient=False,
    )
    print(mod_1.weights)
    print(mod_1.predictions)
    print(mod_1.predict(new_experts=experts.iloc[100:]))
    mod_1.plot_mixture()


    # Example 2
    import pandas as pd
    import numpy as np
    from opera import Mixture

    targets = pd.read_csv("data/targets.csv")["x"]
    experts = pd.read_csv("data/experts.csv")
    awake = np.tile(np.array([1, 0, 1]), (experts.shape[0],)).reshape(experts.shape)

    mod_2 = Mixture(
        y=targets.iloc[0:50],
        experts=experts.iloc[0:50],
        awake=awake[0:50],
        model="BOA",
        loss_type="mse",
        loss_gradient=False,
    )

    mod_2.update(
        new_experts=experts.iloc[50:100], new_y=targets.iloc[50:100], awake=awake[50:100]
    )

    print(mod_2.weights)
    print(mod_2.predictions)
    print(mod_2.predict(new_experts=experts.iloc[100:]))

    # Example 3
    import pandas as pd
    import numpy as np
    from opera import Mixture
    import matplotlib.cm as cm

    targets = pd.read_csv("data/targets.csv")["x"]
    experts = pd.read_csv("data/experts.csv")

    mod_3 = Mixture(
        y=targets.iloc[0:100],
        experts=experts.iloc[0:100],
        model="MLprod",
        loss_type="mse",
        loss_gradient=True,
    )
    print(mod_3.weights)
    print(mod_3.predictions)
    print(mod_3.predict(new_experts=experts.iloc[100:]))
    colors = cm.rainbow(np.linspace(0, 400, 1000))
    mod_3.plot_mixture(plot_type="plot_weight", colors=colors, title = "Custom title", ylabel = "Ylabel")

    # Example 4
    import pandas as pd
    from opera.mixture import Mixture
    import numpy as np

    targets = pd.read_csv("data/targets.csv")["x"]
    experts = pd.read_csv("data/experts.csv")
    N = experts.shape[1]
    w0 = np.full(N, 1 / N)
    fun_reg = lambda x: sum(x * np.log(x / w0))
    fun_reg_grad = lambda x: np.log(x / w0) + 1
    constraints = []
    eq_constraints = {
        "type": "eq",
        "fun": lambda x: sum(x) -1,
        "jac": lambda x: np.ones((1, N)),
    }
    constraints.append(eq_constraints)
    ineq_constraints = {
        "type": "ineq",
        "fun": lambda x: x,
        "jac": lambda x: np.eye(N),
    }
    constraints.append(ineq_constraints)
    parameters = {
        "fun_reg":fun_reg,
        "fun_reg_grad":fun_reg_grad,
        "constraints":constraints,
        "tol":1e-20,
        "options":{"maxiter":50},
    }
    mod_1 = Mixture(
        y=targets,
        experts=experts,
        model="FTRL",
        loss_type="mse",
        loss_gradient=True,
        parameters=parameters
    )
    """

    def __init__(
        self,
        y,
        experts,
        awake=None,
        model="BOA",
        coefficients="uniform",
        loss_type="mse",
        loss_gradient=True,
        parameters=None,
    ):
        if callable(loss_type):
            self.loss_type = loss_type
            self.loss_function = loss_type
            if loss_gradient and not callable(loss_gradient):
                raise ValueError(
                    "When a custom loss function is passed the loss_gradient should be either False or the gradient function corresponding to the loss function"
                )
            if callable(loss_gradient):
                self.loss_type = loss_gradient
        elif loss_type.lower() in ["mape", "mae", "mse", "msle", "mspe"]:
            self.loss_function = globals()[loss_type.lower()]
            if loss_gradient and not callable(loss_gradient):
                self.loss_type = globals()["gradient_" + loss_type.lower()]
            elif loss_gradient and callable(loss_gradient):
                self.loss_type = loss_gradient
            else:
                self.loss_type = globals()[loss_type.lower()]
        else:
            raise NotImplementedError(f"{loss_type} loss function is not implemented.")
        self.model = model
        self.loss_gradient = loss_gradient
        self.gradient_to_call = getattr(self, "r_by_hand")

        if not isinstance(experts, pd.DataFrame):
            raise (TypeError("Experts must be a pandas dataframe"))

        self.experts_names = experts.columns

        batch_shape = experts.shape[:-1]
        self.K = experts.shape[-1]
        weights_shape = [1] * (len(batch_shape) - 1) + [self.K]
        # Initialize variables
        if coefficients == "uniform":
            self.w = np.full(experts.shape[-1], 1 / experts.shape[-1])
        elif isinstance(coefficients, np.ndarray):
            if coefficients.shape[0] != experts.shape[-1]:
                raise ValueError(
                    f"Bad dimention for coefficients, expected {experts.shape[-1]} got {coefficients.shape[0]}"
                )
            self.w = coefficients
        else:
            raise ValueError(
                f'Wrong value for coefficients, expected an np.ndarray of shape {experts.shape[-1]} or "uniform"'
            )
        self.cum_vars = np.ones(weights_shape) / np.power(2, 20)
        self.max_losses = np.ones(weights_shape) / np.power(2, 20)
        self.cum_regrets = np.zeros(weights_shape)
        self.cum_reg_regrets = np.zeros(weights_shape)
        self.learning_rates = np.ones(weights_shape) / np.power(2, 20)
        self.max_sq_regrets = np.zeros(weights_shape)
        self.predictions = np.array([])
        self.weights = np.empty((0, experts.shape[-1]))
        self.awakes = np.empty((0, experts.shape[-1]))
        self.experts = np.empty((0, experts.shape[-1]))
        self.targets = np.array([])
        self.N = experts.shape[-1]
        if model.upper() == "BOA":
            self.predict_at_t = getattr(self, "predict_at_t_BOA")
            self.update_coefficient = getattr(self, "update_coefficient_BOA")
        elif model.upper() == "MLPOL":
            self.predict_at_t = getattr(self, "predict_at_t_MLPol")
            self.update_coefficient = getattr(self, "update_coefficient_MLPol")
        elif model.upper() == "MLPROD":
            self.predict_at_t = getattr(self, "predict_at_t_MLProd")
            self.update_coefficient = getattr(self, "update_coefficient_MLProd")
        elif model.upper() == "FTRL":
            self.predict_at_t = getattr(self, "predict_at_t_FTRL")
            self.update_coefficient = getattr(self, "update_coefficient_FTRL")
            if (
                loss_gradient is not None
                and not callable(loss_gradient)
                and loss_gradient is False
            ):
                raise ValueError(
                    "loss_gradient must be provided to use the FTRL algorithm."
                )
            self.eta = True
            self.eta = float("inf")
            self.default_eta = True
            if parameters is not None:
                self.tol = parameters["tol"] if "tol" in parameters else 1e-20
                self.options = (
                    parameters["options"] if "options" in parameters else None
                )
            else:
                self.tol = 1e-20
                self.options = None
            self.N = experts.shape[1]  # Number of experts
            self.T = experts.shape[0]  # Number of instants
            self.w0 = np.full(self.N, 1 / self.N)
            if (
                parameters is None
                or "fun_reg" not in parameters
                or parameters["fun_reg"] is None
            ):
                self.fun_reg = lambda x: sum(x * np.log(x / self.w0))
                self.fun_reg_grad = lambda x: np.log(x / self.w0) + 1
                self.constraints = []
                eq_constraints = {
                    "type": "eq",
                    "fun": lambda x: sum(x) - 1,
                    "jac": lambda x: np.ones((1, self.N)),
                }
                self.constraints.append(eq_constraints)
                ineq_constraints = {
                    "type": "ineq",
                    "fun": lambda x: x,
                    "jac": lambda x: np.eye(self.N),
                }
                self.constraints.append(ineq_constraints)
            else:
                self.fun_reg = parameters["fun_reg"]
                self.fun_reg_grad = (
                    parameters["fun_reg_grad"] if "fun_reg_grad" in parameters else None
                )
                self.constraints = (
                    parameters["constraints"] if "constraints" in parameters else None
                )

            self.G = np.zeros(self.N)
            # Define the initial values of the variables
            x0 = [1 / self.N for i in range(self.N)]

            result = minimize(
                self.fun_reg,
                x0,
                method="SLSQP",
                constraints=self.constraints,
                tol=self.tol,
                options=self.options,
                jac=self.fun_reg_grad,
            )
            self.weights = np.vstack((self.weights, result.x))
        else:
            raise NotImplementedError(f"Algorithm {model} is not implemented.")
        self.update(experts, y, awake=awake)

    def r_by_hand(self, x, y, awake=None):
        """Compute the gradient of the loss function with respect to the weights."""
        batch_shape = x.shape[:-1]
        batch_axes = list(range(len(batch_shape)))
        y_hat = np.sum(self.w * x, axis=-1, keepdims=True)
        if not self.loss_gradient:
            r = awake * (self.loss_type(y_hat, y) - self.loss_type(x, y))
        else:
            r = awake * (
                self.loss_type(y_hat, y) * y_hat - self.loss_type(y_hat, y) * x
            )

        r = np.mean(r, axis=tuple(batch_axes))
        return y_hat, r

    def predict(self, new_experts, awake=None):
        """Performs sequential predictions and updates of a mixture object based on new observations and last coefficients
        Args:
            new_experts (numpy.array or pandas.Dataframe): an array of new experts.
            awake (numpy.array or pandas.Dataframe, optional): an array specifying the activation coefficients of the experts.
                It must have the same dimension as experts. Defaults to None.
        Returns:
            numpy.array: array of predictions based on the new experts and last coefficients
        """
        new_experts = self.check_columns(new_experts)
        awake = self.check_awake(awake=awake, x=new_experts)
        x = new_experts.to_numpy()
        coef = np.apply_along_axis(normalize, 1, (awake * self.w))
        y_hat = np.sum(coef * x, axis=-1, keepdims=True)
        return y_hat

    def check_columns(self, experts):
        if set(experts.columns) != set(self.experts_names):
            raise (
                ValueError(
                    f"Bad experts columns, expected {list(self.experts_names)} found {list(experts.columns)}"
                )
            )
        return experts[self.experts_names]

    def check_awake(self, awake, x):
        if awake is None:
            awake = np.ones(x.shape)
        if awake.shape != x.shape:
            raise ValueError(
                f"Bad dimention for awake, expexted {x.shape} got {awake.shape}"
            )
        if isinstance(awake, pd.DataFrame):
            if set(awake.columns) != set(self.experts_names):
                raise (
                    ValueError(
                        f"Bad awake columns, expected {list(self.experts_names)} found {list(awake.columns)}"
                    )
                )
            awake = awake[self.experts_names]
            awake = awake.to_numpy()
        return awake

    def update(self, new_experts, new_y, awake=None):
        """updates the model sequentially with new experts and new targets

        Args:
            new_experts (numpy.array or pandas.DataFrame): matrix of nex experts used to update the model
            new_y (numpy.array or pandas.DataFrame): array of new targets used to update the model
            awake (numpy.array or pandas.Dataframe, optional): an array specifying the activation coefficients
                of the experts. It must have the same dimension as experts. Defaults to None.
        """
        new_experts = self.check_columns(new_experts)
        awake = self.check_awake(awake, new_experts)
        if not isinstance(new_experts, np.ndarray):
            x = new_experts.to_numpy()
        else:
            x = new_experts
        if not isinstance(new_y, np.ndarray):
            y = new_y.to_numpy()
        else:
            y = new_y
        if not isinstance(awake, np.ndarray):
            awake = awake.to_numpy()
        if x.shape[:-1] != y.shape:
            raise ValueError("Bad dimensions: x and y should have the same shape")
        for index, value in enumerate(y):
            xt = x[index]
            yt = np.expand_dims(value, -1)
            y_hat, updates = self.predict_at_t(xt, yt, awake=awake[index, :])
            self.predictions = np.append(self.predictions, y_hat)
            self.weights = np.vstack((self.weights, updates.get("weights")))
            self.experts = np.vstack((self.experts, xt))
            self.targets = np.append(self.targets, value)
        self.awakes = np.vstack((self.awakes, awake))

        self.loss = np.mean(self.loss_function(self.predictions, self.targets))
        self.update_coefficient()

    def predict_at_t_BOA(self, x, y, awake=None):
        """predicts at time t using BOA."""
        idx = awake > 0
        Raux = (
            np.log(self.learning_rates)
            + np.log(1 / self.K)
            + self.learning_rates * self.cum_reg_regrets
        )
        Rmax = np.max(Raux[idx])
        self.w = np.zeros(self.N)
        self.w[idx] = np.exp(Raux[idx] - Rmax)
        self.w = normalize(self.w)
        y_hat, r = self.gradient_to_call(x, y, awake=awake)
        r_square = np.square(r)
        self.max_losses = np.maximum(self.max_losses, np.abs(r))  # 64
        B2 = np.power(2, np.ceil(np.log2(self.max_losses)))
        self.cum_vars += r_square
        self.learning_rates = np.minimum(
            1 / B2, np.sqrt(np.log(self.K) / self.cum_vars)
        )
        self.cum_reg_regrets += (
            1
            / 2
            * (
                r
                - self.learning_rates * r_square
                + B2 * (self.learning_rates * r > 1 / 2)
            )
        )
        self.cum_regrets += r

        slot_variables_updates = {
            "cum_vars": self.cum_vars,
            "max_losses": self.max_losses,
            "learning_rates": self.learning_rates,
            "cum_regrets": self.cum_regrets,
            "weights": self.w,
        }

        return y_hat, slot_variables_updates

    def update_coefficient_BOA(self):
        Raux = (
            np.log(self.learning_rates)
            + np.log(1 / self.K)
            + self.learning_rates * self.cum_reg_regrets
        )
        Rmax = np.max(Raux)
        self.w = np.zeros(self.N)
        self.w = np.exp(Raux - Rmax)
        self.w = normalize(self.w)

    def predict_at_t_MLPol(self, x, y, awake=None):
        """predicts at time t using MLpol."""
        np_relu = np.maximum(self.cum_regrets, 0)
        self.w = np.multiply(self.learning_rates, np_relu)
        w_sum = np.sum(self.w, axis=-1, keepdims=True)
        if w_sum != 0:
            self.w = np.where(
                np.equal(w_sum, np.zeros(self.w.shape)),
                np.ones(self.w.shape) / self.K,
                np.divide(self.w, w_sum),
            )
        else :
            self.w = np.ones(self.w.shape) / self.K

        self.w = normalize(awake * self.w)

        y_hat, r = self.gradient_to_call(x, y, awake=awake)

        r_square = np.square(r)
        self.cum_regrets += r

        max_squared_regret_diff = np.maximum(
            np.max(r_square, axis=-1, keepdims=True) - self.max_sq_regrets, 0
        )

        self.learning_rates = 1 / (
            1 / self.learning_rates + r_square + max_squared_regret_diff
        )
        self.max_sq_regrets += max_squared_regret_diff
        slot_variables_updates = {
            "max_squared_regret": self.max_sq_regrets,
            "learning_rates": self.learning_rates,
            "cum_regrets": self.cum_regrets,
            "weights": self.w,
        }

        return y_hat, slot_variables_updates

    def update_coefficient_MLPol(self):
        np_relu = np.maximum(self.cum_regrets, 0)
        self.w = np.multiply(self.learning_rates, np_relu)
        w_sum = np.sum(self.w, axis=-1, keepdims=True)
        if w_sum != 0:
            self.w = np.where(
                np.equal(w_sum, np.zeros(self.w.shape)),
                np.ones(self.w.shape) / self.K,
                np.divide(self.w, w_sum),
            )
        else :
            self.w = np.ones(self.w.shape) / self.K
        self.w = normalize(self.w)

    def predict_at_t_MLProd(self, x, y, awake=None):
        """predicts at time t using MLprod."""
        self.w = np.multiply(self.learning_rates, np.exp(self.cum_regrets))
        self.w = np.divide(self.w, np.sum(self.w, axis=-1, keepdims=True))
        self.w = normalize(awake * self.w)

        y_hat, r = self.gradient_to_call(x, y, awake=awake)
        r_square = np.square(r)
        self.cum_vars += r_square
        self.max_losses = np.maximum(self.max_losses, np.abs(r))
        epsilon = 1e-30
        new_learning_rates = np.minimum(
            np.minimum(0.5 / self.max_losses, np.sqrt(np.log(self.K) / self.cum_vars)),
            np.ones(self.learning_rates.shape) / epsilon,
        )
        self.cum_regrets = (
            new_learning_rates / self.learning_rates * self.cum_regrets
            + np.log(1 + new_learning_rates * r)
        )
        self.learning_rates = new_learning_rates

        slot_variables_updates = {
            "cum_vars": self.cum_vars,
            "max_losses": self.max_losses,
            "learning_rates": self.learning_rates,
            "cum_regrets": self.cum_regrets,
            "weights": self.w,
        }

        return y_hat, slot_variables_updates

    def update_coefficient_MLProd(self):
        self.w = np.multiply(self.learning_rates, np.exp(self.cum_regrets))
        self.w = np.divide(self.w, np.sum(self.w, axis=-1, keepdims=True))
        self.w = normalize(self.w)

    def predict_at_t_FTRL(self, x, y, awake=None):
        y_hat = np.sum(self.weights[-1] * x)
        G_t = self.loss_type(y_hat, y) * x
        self.G = self.G + G_t
        if self.default_eta:
            self.eta = 1 / np.sqrt(1 / np.square(self.eta) + np.sum(np.square(G_t)))
        obj = lambda x: (self.fun_reg(x) + self.eta * sum(self.G * x))
        obj_grad = (
            None
            if self.fun_reg_grad is None
            else lambda x: self.fun_reg_grad(x) + self.eta * self.G
        )

        x0 = self.w
        result = minimize(
            obj,
            x0,
            method="SLSQP",
            constraints=self.constraints,
            tol=self.tol,
            options=self.options,
            jac=obj_grad,
        )
        self.w = result.x
        slot_variables_updates = {
            "weights": self.w,
        }

        return y_hat, slot_variables_updates

    def update_coefficient_FTRL(self):
        self.weights = np.delete(self.weights, -1, axis=0)

    def plot_mixture(
        self,
        plot_type="all",
        colors=None,
        max_experts=None,
        title=None,
        ylabel=None,
        index_start=None,
        index_stop=None,
    ):
        """provides different diagnostic plots for an aggregation procedure.

        Args:
            plot_type (str, optional): string specifying the plots to show, available plots:
                - plot_weight: Weights associated with the experts
                - boxplot_weight: weights associated with the experts
                - dyn_avg_loss: dynamic average loss
                - cumul_res: cumulative Residuals
                - avg_loss: average loss suffered by the experts
                - contrib: contibution of each expert to the prediction
                - all display all the above graphs
                Defaults to "all".
            colors (numpy.array, optional): array of colors to be used for the plots. Defaults to None.
            max_experts (int): max number of expert to be displayed
            title (str, optional) : title. Only available plotting one graphic (not using plot_type = "all")
            ylabel (str, optional) : ylabel. Only available plotting one graphic (not using plot_type = "all")
            index_start (int) : the index where the plots starts (may be positive or negative)
            index_stop (int) : the index where the plots stops (may be positive or negative)

        Examples :

        import pandas as pd
        from opera.mixture import Mixture
        import numpy as np
        targets = pd.read_csv("data/targets.csv")["x"]
        experts = pd.read_csv("data/experts.csv")
        mod_1 = Mixture(y=targets,experts=experts,model="BOA",loss_type="mse",loss_gradient=True)

        # default plot
        mod_1.plot_mixture()

        # plot one
        mod_1.plot_mixture(plot_type="contrib")

        # plot only last 100 predictions
        mod_1.plot_mixture(index_start=-100)

        # plot only first 50 predictions
        mod_1.plot_mixture(index_stop=-50)

        # plot between two indexes
        mod_1.plot_mixture(index_start=10,index_stop=50)

        # plot with custom colors
        import seaborn as sns
        colors = sns.color_palette("Set2", experts.shape[0] + 2)
        mod_1.plot_mixture(colors=colors)

        """
        figsize = (10, 8)
        K = self.experts.shape[1]

        if colors is None:
            colors = sns.color_palette(None, K + 2)
        colors = np.array(colors)
        if not max_experts or max_experts > K:
            max_experts = K
        labels = np.array(self.experts_names)
        if plot_type == "all":
            fig, ax = plt.subplots(3, 2, figsize=figsize, dpi=100)
            # Stack plot of weights associated to each expert
            plot_weight(
                ax[0, 0],
                colors,
                self,
                max_experts,
                index_start=index_start,
                index_stop=index_stop,
            )
            # Boxplot of weights associated to each expert
            boxplot_weight(
                ax[0, 1],
                colors,
                self,
                max_experts,
                index_start=index_start,
                index_stop=index_stop,
            )
            # Barplot loss
            dyn_avg_loss(
                ax[1, 0],
                colors,
                self,
                max_experts,
                index_start=index_start,
                index_stop=index_stop,
            )
            # Cumulative residuals
            cumul_res(
                ax[1, 1],
                colors,
                self,
                max_experts,
                index_start=index_start,
                index_stop=index_stop,
            )

            # Average loss
            avg_loss(
                ax[2, 0],
                colors,
                self,
                max_experts,
                index_start=index_start,
                index_stop=index_stop,
            )

            # Average loss
            contrib(
                ax[2, 1],
                colors,
                self,
                max_experts,
                index_start=index_start,
                index_stop=index_stop,
            )

            handles, labels = ax[1, 1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=10, bbox_to_anchor=(0.5, 1), frameon=False)
            fig.suptitle(" ", fontsize=24)
            fig.tight_layout()
        elif plot_type == "boxplot_weight":
            fig, ax = plt.subplots(dpi=100)
            # Boxplot of weights associated to each expert
            handles = boxplot_weight(
                ax,
                colors,
                self,
                max_experts,
                title,
                ylabel,
                index_start=index_start,
                index_stop=index_stop,
            )
            fig.suptitle(" ", fontsize=16)
            fig.tight_layout()
        elif plot_type == "plot_weight":
            fig, ax = plt.subplots(dpi=100)
            # Stack plot of weights associated to each expert
            plot_weight(
                ax,
                colors,
                self,
                max_experts,
                title,
                ylabel,
                index_start=index_start,
                index_stop=index_stop,
            )
            fig.legend(loc="upper center", ncol=K + 2, borderaxespad=1.0, bbox_to_anchor=(0.5, 1), frameon=False)
            fig.suptitle(" ", fontsize=16)
            fig.tight_layout()
        elif plot_type == "contrib":
            fig, ax = plt.subplots(dpi=100)
            # Stack plot of weights associated to each expert
            contrib(
                ax,
                colors,
                self,
                max_experts,
                title,
                ylabel,
                index_start=index_start,
                index_stop=index_stop,
            )
            fig.legend(loc="upper center", ncol=K + 2, bbox_to_anchor=(0.5, 1), frameon=False)
            fig.suptitle(" ", fontsize=16)
            fig.tight_layout()
        elif plot_type == "dyn_avg_loss":
            fig, ax = plt.subplots(dpi=100)
            # Barplot loss
            dyn_avg_loss(
                ax,
                colors,
                self,
                max_experts,
                title,
                ylabel,
                index_start=index_start,
                index_stop=index_stop,
            )
            fig.legend(loc="upper center", ncol=K + 2, bbox_to_anchor=(0.5, 1), frameon=False)
            fig.suptitle(" ", fontsize=16)
            fig.tight_layout()
        elif plot_type == "cumul_res":
            fig, ax = plt.subplots(dpi=100)
            # Cumulative residuals
            cumul_res(
                ax,
                colors,
                self,
                max_experts,
                title,
                ylabel,
                index_start=index_start,
                index_stop=index_stop,
            )
            fig.legend(loc="upper center", ncol=K + 2, bbox_to_anchor=(0.5, 1), frameon=False)
            fig.suptitle(" ", fontsize=16)
            fig.tight_layout()
        elif plot_type == "avg_loss":
            fig, ax = plt.subplots(dpi=100)
            # Average loss
            avg_loss(
                ax,
                colors,
                self,
                max_experts,
                title,
                ylabel,
                index_start=index_start,
                index_stop=index_stop,
            )
            fig.suptitle(" ", fontsize=16)
            fig.tight_layout()
        else:
            raise (NotImplementedError(f"{plot_type} plot not implemented yet."))
        plt.show()
