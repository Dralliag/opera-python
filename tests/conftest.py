import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def init():
    n = 50
    experts = pd.DataFrame({"Exp1": [0] * n, "Exp2": [1] * n})
    experts.iloc[n - 1].Exp1 = 1
    y = np.repeat(0.4, n)
    y[n - 1] = 1
    awakes = pd.DataFrame({"Exp1": [0, 1] * (n // 2), "Exp2": [1] * n})

    return experts, y, awakes
