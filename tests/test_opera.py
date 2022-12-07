import pandas as pd
from opera import Mixture
import numpy as np
from plot_mixture import plot_mixture

targets = pd.read_csv("data/targets.csv")["x"]
experts = pd.read_csv("data/experts.csv")

awake = np.tile(np.array([1, 0, 1]), (experts.shape[0],)).reshape(experts.shape)

# model  sur les 100 premiers
mod_1 = Mixture(
    y=targets.loc[0:99],
    experts=experts.loc[0:99],
    awake=awake[0:100],
    model="BOA",
    loss_type="mse",
    loss_gradient=False,
    epsilon=1e-30,
)

# iso, mais avec init + update
mod_2 = Mixture(
    y=targets.loc[0:49],
    experts=experts.loc[0:49],
    awake=awake[0:50],
    model="BOA",
    loss_type="mse",
    loss_gradient=False,
    epsilon=1e-30,
)

mod_2.update(new_experts=experts.loc[50:99], new_y=targets.loc[50:99], awake=awake[50:100])

# TO DO : format simplifie des weights / predictions, ...
pd.DataFrame(mod_1.weights)
pd.DataFrame(mod_2.weights)

pd.DataFrame(mod_1.predictions)
pd.DataFrame(mod_2.predictions)

# predict
mod_1.predict(new_experts=experts.loc[100:])
mod_2.predict(new_experts=experts.loc[100:])

plot_mixture(mod_1)
s
