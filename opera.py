"""
opyra - Online Python by Expert Aggregation
"""

# Author : Carl Remlinger <carl.remlinger@epfl.ch>
# Largely inspired by the work of: Opera R-package by
#   - Pierre Gaillard
#   - Yanning Goude
# License: MIT


import numpy as np  # only needed for isintance l176
import pandas as pd  # only needed for isintance l171

# Losses
def mape(x, y):
    return np.abs(y - x) / x


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


def mspe(x, y):
    return np.square(y - x) / np.square(x)


def normalize(x):
    return x / sum(x)


class Mixture:
    def __init__(
        self,
        y,
        experts,
        awake=None,
        model="BOA",
        coefficients="uniform",
        loss_type=mse,
        loss_gradient=True,
        epsilon=1e-30,
    ):
        if callable(loss_type):
            self.loss_type = loss_type
            if loss_gradient and not callable(loss_gradient):
                raise ValueError(
                    "When a custom loss function is passed the loss_gradient should be either False or the gradient function corresponding to the loss function"
                )
        elif loss_type.lower() in ["mape", "mae", "mse", "msle", "mspe"]:
            if loss_gradient:
                self.loss_type = globals()["gradient_" + loss_type.lower()]
                print("gradient_" + loss_type.lower())
            else:
                self.loss_type = globals()[loss_type.lower()]
        else:
            raise NotImplementedError(f"{loss_type} loss function is not implemented.")
        self.epsilon = epsilon
        self.model = model
        self.gradient_to_call = getattr(self, "r_by_hand")
        if model.upper() == "BOA":
            self.predict_at_t = getattr(self, "predict_at_t_BOA")
        elif model.upper() == "MLPOL":
            self.predict_at_t = getattr(self, "predict_at_t_MLPol")
        elif model.upper() == "MLPROD":
            self.predict_at_t = getattr(self, "predict_at_t_MLProd")
        else:
            raise NotImplementedError(f"Algorithm {model} is not implemented.")

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
        else:
            raise ValueError(
                f'Wrong value for coefficients, expected an np.ndarray of shape {experts.shape[-1]} or "uniform"'
            )
        self.cum_vars = np.zeros(weights_shape)
        self.max_losses = np.zeros(weights_shape)
        self.cum_regrets = np.zeros(weights_shape)
        self.cum_reg_regrets = np.zeros(weights_shape)
        # self.learning_rates = np.ones(weights_shape) / self.epsilon
        self.learning_rates = np.ones(weights_shape)
        self.max_sq_regrets = np.zeros(weights_shape)
        self.predictions = []
        self.weights = []
        self.awakes = []
        self.experts = []
        self.targets = []
        self.N = experts.shape[-1]
        self.update(experts, y, awake=awake)

    def r_by_hand(self, x, y, awake=None):
        """
        Compute the gradient of the loss function with respect to the weights.

        Parameters
        ----------
        x : {tensor, sparse matrix} of (n_samples, n_dim, n_experts)
            Experts.
        y : {tensor, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_dim)
            Target.
        
        Returns
        -------
        y_hat : {tensor, sparse matrix} of shape (n_samples,) or \
        r : {tensor, sparse matrix} of shape (n_samples, n_experts)
        """
        batch_shape = x.shape[:-1]
        batch_axes = list(range(len(batch_shape)))
        y_hat = np.sum(self.w * x, axis=-1, keepdims=True)
        r = awake * (self.loss_type(y_hat, y) - self.loss_type(x, y))
        self.awakes.append(awake)
        r = np.mean(r, axis=tuple(batch_axes))
        return y_hat, r

    def predict(self, new_experts, awake=None):
        new_experts = self.check_columns(new_experts)
        awake = self.check_awake(awake=awake, x=new_experts)
        x = new_experts.to_numpy()
        coef = (awake * self.w) / np.sum(awake * self.w, axis=1, keepdims=True)
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
        return awake

    def update(self, new_experts, new_y, awake=None):
        new_experts = self.check_columns(new_experts)
        awake = self.check_awake(awake, new_experts)
        x = new_experts.to_numpy()
        y = new_y.to_numpy()
        if x.shape[:-1] != y.shape:
            raise ValueError("Bad dimensions: x and y should have the same shape")
        for index, value in enumerate(y):
            xt = x[index]
            yt = np.expand_dims(value, -1)
            y_hat, updates = self.predict_at_t(xt, yt, awake=awake[index, :])
            self.predictions.append(np.squeeze(y_hat, axis=-1))
            self.weights.append(updates.get("weights"))
            self.experts.append(xt)
            self.targets.append(value)
        # return self.predictions

    def predict_at_t_BOA(self, x, y, awake=None):
        """
        Predict at time t.
        
        Parameters
        ----------
        
        x : {tensor, sparse matrix} of (n_samples, n_dim, n_experts)
            Experts.
        y : {tensor, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_dim)
            Target.
        
        Returns
        -------
        y_hat : {tensor, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_dim)
            Predictions.
        
        slot_variables_updates : dict
            Dictionary of updates for the slot variables.
        """
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
        }  # could be compute with learning_rates and cum_regrets

        return y_hat, slot_variables_updates

    def predict_at_t_MLPol(self, x, y, awake=None):
        np_relu = np.maximum(self.cum_regrets, 0)
        self.w = np.multiply(self.learning_rates, np_relu)
        w_sum = np.sum(self.w, axis=-1, keepdims=True)
        self.w = np.where(
            np.equal(w_sum, np.zeros(self.w.shape)),
            np.ones(self.w.shape) / self.K,
            np.divide(self.w, w_sum),
        )
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

    def predict_at_t_MLProd(self, x, y, awake=None):

        self.w = np.multiply(self.learning_rates, np.exp(self.cum_regrets))
        self.w = np.divide(self.w, np.sum(self.w, axis=-1, keepdims=True))
        self.w = normalize(awake * self.w)

        y_hat, r = self.gradient_to_call(x, y, awake=awake)
        r_square = np.square(r)
        self.cum_vars += r_square
        self.max_losses = np.maximum(self.max_losses, np.abs(r))

        new_learning_rates = np.minimum(
            np.minimum(0.5 / self.max_losses, np.sqrt(np.log(self.K) / self.cum_vars)),
            np.ones(self.learning_rates.shape) / self.epsilon,
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
