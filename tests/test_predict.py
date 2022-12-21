import pytest
from opera.mixture import Mixture
import numpy as np


def mape(x, y):
    return np.abs(x - y) / y


def mae(x, y):
    return np.abs(x - y)


def mse(x, y):
    return np.square(x - y)


def one_model(init, model):
    experts, y, awakes = init
    for possible_loss in ["mape", "mae", "mse"]:
        m = Mixture(
            experts=experts[:-10],
            y=y[:-10],
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        m1 = Mixture(
            experts=experts[:1],
            y=y[:1],
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        for i in range(1, experts.shape[0] - 10):
            m1.update(new_experts=experts[i : i + 1], new_y=y[i : i + 1])

        assert (m.weights == m1.weights).all()
        assert (m.predictions == m1.predictions).all()
        assert (m.w == m1.w).all()
        assert m.loss == m1.loss
        pred_m = m.predict(experts[-10:])
        pred_m1 = m1.predict(experts[-10:])
        assert (pred_m == pred_m1).all()
        # with awakes

        m = Mixture(
            experts=experts[:-10],
            y=y[:-10],
            awake=awakes[:-10],
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        m1 = Mixture(
            experts=experts[:1],
            y=y[:1],
            awake=awakes[:1],
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        for i in range(1, experts.shape[0] - 10):
            m1.update(
                new_experts=experts[i : i + 1],
                new_y=y[i : i + 1],
                awake=awakes[i : i + 1],
            )

        assert (m.weights == m1.weights).all()
        assert (m.predictions == m1.predictions).all()
        assert (m.w == m1.w).all()
        assert m.loss == m1.loss
        pred_m = m.predict(experts[-10:], awake=awakes[-10:])
        pred_m1 = m1.predict(experts[-10:], awake=awakes[-10:])
        assert (pred_m == pred_m1).all()


def test_MLpol(init):
    one_model(init, "MLpol")


def test_BOA(init):
    one_model(init, "BOA")


def test_MLprod(init):
    one_model(init, "MLprod")
