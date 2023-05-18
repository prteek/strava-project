import matplotlib.pyplot as plt
import numpy as np
import awswrangler as wr
import pandas as pd
import torch
import torch.nn as nn
from helpers import (
    TARGET_FITNESS as TARGET,
    PREDICTORS_FITNESS,
    dtype_converter
)
from sklearn.preprocessing import StandardScaler, FunctionTransformer as FT
from sklearn.pipeline import Pipeline
from skorch import NeuralNetRegressor
import os

os.environ["AWS_PROFILE"] = "personal"

training_dir = "./data"
df_train = (
    pd.read_csv(os.path.join(training_dir, "train.csv"))
    .dropna(subset=[*PREDICTORS_FITNESS, *TARGET])
    .reset_index(drop=True)
)

X = df_train[PREDICTORS_FITNESS]
y = df_train[TARGET].values.astype(np.float32)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(3, 50)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Dropout(0.2)
        self.layer4 = nn.Linear(50, 50)
        self.layer5 = nn.ReLU()
        self.layer6 = nn.Dropout(0.2)
        self.layer7 = nn.Linear(50, 50)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Dropout(0.2)
        self.layer10 = nn.Linear(50, 50)
        self.layer11 = nn.ReLU()
        self.layer12 = nn.Dropout(0.2)
        self.layer13 = nn.Linear(50, 50)
        self.layer14 = nn.ReLU()
        self.layer15 = nn.Dropout(0.2)
        self.layer16 = nn.Linear(50, 50)
        self.layer17 = nn.ReLU()
        self.layer18 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        return x

def __call__(self):
    for module in self.modules:
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0.0, None)
            module.weight.data = w


estimator = NeuralNetRegressor(
    NeuralNet,
    max_epochs=1000,
    criterion=nn.MSELoss(),
    lr=0.01,
    # Shuffle training data on each epoch
    iterator_train__shuffle=False,
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("dtype_converter", FT(dtype_converter)),
        ("estimator", estimator),
    ]
)


model.fit(X,y)

d = np.arange(100)
ini = np.ones_like(d)*30
ss = np.ones_like(d)*0
x = pd.DataFrame(np.c_[ini, d, ss], columns=PREDICTORS_FITNESS)
plt.plot(d, model.predict(x)[:,0])
plt.plot(d, ini*np.exp(-0.028*d))
# plt.scatter([0,35], [6.2,2.3])
