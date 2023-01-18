import pandas as pd
from opera.mixture import Mixture
import numpy as np

targets = pd.read_csv("data/targets.csv")["x"]
experts = pd.read_csv("data/experts.csv")
model = "FTRL"

# FTRL with default parameters
mod_1 = Mixture(
    y=targets.iloc[0:100],
    experts=experts.iloc[0:100],
    model=model,
    loss_type="mse",
    loss_gradient=True,
)

print(mod_1.w)


# FTRL with custom parameters
N = experts.shape[1]
w0 = np.full(N, 1 / N)
fun_reg = lambda x: sum(x * np.log(x / w0))
fun_reg_grad = lambda x: np.log(x / w0) + 1
constraints = []
eq_constraints = {
    "type": "eq",
    "fun": lambda x: sum(x) - 1,
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
    "fun_reg": fun_reg,
    "fun_reg_grad": fun_reg_grad,
    "constraints": constraints,
    "tol": 1e-20,
    "options": {"maxiter": 50},
}
mod_2 = Mixture(
    y=targets.iloc[0:100],
    experts=experts.iloc[0:100],
    model=model,
    loss_type="mse",
    loss_gradient=True,
    parameters=parameters,
)
print(mod_2.w)

print((mod_1.w == mod_2.w).all())
