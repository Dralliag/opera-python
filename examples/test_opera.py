import pandas as pd
from opera.mixture import Mixture
import numpy as np

# Documentation
?Mixture
?Mixture.plot_mixture

targets = pd.read_csv("data/targets.csv")["x"]
experts = pd.read_csv("data/experts.csv")

# model = "BOA"
# model = "MLpol"
model = "MLprod"

set_awake = True

if set_awake:
  awake = np.tile(np.array([1, 0, 1]), (experts.shape[0],)).reshape(experts.shape)
else:
  awake = np.tile(np.array([1, 1, 1]), (experts.shape[0],)).reshape(experts.shape)

# model sur les 100 premiers
mod_1 = Mixture(
    y=targets.iloc[0:100],
    experts=experts.iloc[0:100],
    awake=awake[0:100],
    model=model,
    loss_type="mse",
    loss_gradient=False,
)

# iso, mais avec init + update
mod_2 = Mixture(
    y=targets.iloc[0:50],
    experts=experts.iloc[0:50],
    awake=awake[0:50],
    model=model,
    loss_type="mse",
    loss_gradient=False,
)

mod_2.update(
    new_experts=experts.iloc[50:100], new_y=targets.iloc[50:100], awake=awake[50:100]
)

print(mod_1.weights)
print(mod_2.weights)

print(mod_1.predictions)
print(mod_2.predictions)

# predict
mod_1.predict(new_experts=experts.iloc[100:], awake=awake[100:])
mod_2.predict(new_experts=experts.iloc[100:], awake=awake[100:])

mod_1.plot_mixture()
