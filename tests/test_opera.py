import os

os.chdir("/home/riad/EDF/opera_py")
import pandas as pd
from opera import Mixture
import numpy as np
from plot_mixture import plot_mixture


targets = pd.read_csv("data/targets.csv")["x"]
experts = pd.read_csv("data/experts.csv")

awake = np.tile(np.array([1, 0, 1]), (experts.shape[0],)).reshape(experts.shape)

agg = Mixture(
    Y=targets,
    experts=experts,
    awake=awake,
    model="BOA",
    loss_type="mse",
    loss_gradient=False,
    epsilon=1e-30,
)

agg.weights

plot_mixture(agg, plot_type="all")
