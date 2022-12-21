import pytest
from opera.mixture import Mixture
import numpy as np


def custom_mape(x, y):
    return np.abs(x - y) / y


def custom_gradient_mape(x, y):
    return 1 / y * np.sign(x - y)


def custom_mae(x, y):
    return np.abs(x - y)


def custom_gradient_mae(x, y):
    return np.sign(x - y)


def custom_mse(x, y):
    return np.square(x - y)


def custom_gradient_mse(x, y):
    return 2 * (x - y)


def one_model(init, model):
    experts, y, _ = init
    for possible_loss in ["mape", "mae", "mse"]:
        custom_loss_type = globals()["custom_" + possible_loss]
        custom_gradient_loss_type = globals()["custom_gradient_" + possible_loss]

        m = Mixture(
            experts=experts,
            y=y,
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )

        m1 = Mixture(
            experts=experts,
            y=y,
            model=model,
            loss_type=custom_loss_type,
            loss_gradient=False,
        )

        assert (m.weights == m1.weights).all()
        assert (m.predictions == m1.predictions).all()
        assert (m.w == m1.w).all()
        assert m.loss == m1.loss

        # with gradient
        m = Mixture(
            experts=experts,
            y=y,
            model=model,
            loss_type=possible_loss,
            loss_gradient=True,
        )

        m1 = Mixture(
            experts=experts,
            y=y,
            model=model,
            loss_type=custom_loss_type,
            loss_gradient=custom_gradient_loss_type,
        )

        assert (m.weights == m1.weights).all()
        assert (m.predictions == m1.predictions).all()
        assert (m.w == m1.w).all()
        assert m.loss == m1.loss


def test_MLpol(init):
    one_model(init, "MLpol")


def test_BOA(init):
    one_model(init, "BOA")


def test_MLprod(init):
    one_model(init, "MLprod")
