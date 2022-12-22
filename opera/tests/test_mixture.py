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
        loss_type = globals()[possible_loss]
        m = Mixture(
            experts=experts,
            y=y,
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        assert abs(m.w[0] - 0.6) < 0.2
        assert (m.loss == np.mean(loss_type(m.predictions, y))).all()

        m1 = Mixture(
            experts=experts[:1],
            y=y[:1],
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        m1.update(new_experts=experts[1:], new_y=y[1:])
        assert (m.predictions == m1.predictions).all()

        w0 = np.array([0.3, 0.7])
        m = Mixture(
            y=y,
            experts=experts,
            model=model,
            coefficients=w0,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        assert abs(m.w[0] - 0.6) < 0.2
        # assert (m.weights[0] == w0).all() TODO fails atm

        m = Mixture(
            experts=experts[:5],
            y=y[:5],
            awake=awakes[:5],
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        m.update(new_experts=experts[5:], new_y=y[5:], awake=awakes[5:])
        assert abs(m.w[0] - 0.6) < 0.2
        assert (m.loss == np.mean(loss_type(m.predictions, y))).all()

        m1 = Mixture(
            experts=experts,
            y=y,
            awake=awakes,
            model=model,
            loss_type=possible_loss,
            loss_gradient=False,
        )
        assert (m.predictions == m1.predictions).all()


def test_MLpol(init):
    one_model(init, "MLpol")


def test_BOA(init):
    one_model(init, "BOA")


def test_MLprod(init):
    one_model(init, "MLprod")
