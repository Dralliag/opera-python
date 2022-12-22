import pytest
from opera.mixture import Mixture


def test_not_implemented_loss(init):
    x, y, _ = init
    with pytest.raises(NotImplementedError):
        Mixture(experts=x, y=y, loss_type="plop")


def test_missing_experts(init):
    _, y, _ = init
    with pytest.raises(TypeError):
        Mixture(y=y)


def test_missing_y(init):
    experts, _, _ = init
    with pytest.raises(TypeError):
        Mixture(experts=experts)


def test_bad_dimensions(init):
    experts, y, awakes = init
    with pytest.raises(ValueError):
        Mixture(experts=experts, y=y[0:10])
    with pytest.raises(ValueError):
        Mixture(experts=experts[0:10], y=y)
    with pytest.raises(ValueError):
        Mixture(experts=experts, y=y, awake=awakes[0:10])
