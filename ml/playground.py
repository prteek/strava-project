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
torch.random.manual_seed(0)


training_dir = "./data"
df_train = (
    pd.read_csv(os.path.join(training_dir, "train.csv"))
    .dropna(subset=[*PREDICTORS_FITNESS, *TARGET])
    .reset_index(drop=True)
)

X = df_train[PREDICTORS_FITNESS]
y = df_train[TARGET].values.astype(np.float32)


class TorchModel(nn.Module):
    def __init__(self, n_units=20, n_hidden=2):
        super(TorchModel, self).__init__()
        self.lin_in = nn.Linear(3,n_units)
        self.hidden = [nn.Linear(n_units,n_units) for i in range(n_hidden)]
        self.lin_out = nn.Linear(n_units,2)

    def forward(self, x):
        x = self._check_types(x)
        x = torch.relu(self.lin_in(x))
        for i_hidden in self.hidden:
            x = torch.relu(i_hidden(x))

        x = self.lin_out(x)
        return x

    @staticmethod
    def _check_types(x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise TypeError(f"Input type {type(x)} not supported")
        return x

    def predict(self, x):
        x = self._check_types(x)
        return self.forward(x).detach().numpy()



model = TorchModel()

mse_loss = torch.nn.MSELoss()


def loss_func(y,y_pred):
    """Compound loss function to apply ODE and boundary conditions to Neural Network"""
    # There is a small precaution in using the gradient with vector input.
    # You should feed the backward function with unit vector in order to access the gradient as a vector.

    # Governing equation constraint
    days = torch.arange(1,11, dtype=torch.float32).requires_grad_(True)
    ini_fit = torch.arange(5,21, dtype=torch.float32).requires_grad_(True)
    x_fit, x_days = torch.meshgrid(ini_fit, days)
    x_ss = torch.zeros_like(x_fit, dtype=torch.float32)
    x_data = torch.concat([x_fit.reshape(-1,1), x_days.reshape(-1,1), x_ss.reshape(-1,1)], dim=1)

    y_ = model(x_data)
    y_fit_pre = y_[:,0]
    y_ini = x_data[:,0]
    dy, = torch.autograd.grad(y_fit_pre, x_data,
                                grad_outputs=torch.ones_like(y_fit_pre),
                                create_graph=True,  # Needed since the ODE function itself is differentiated further
                                # in training loop making this step twice differentiable,
                                allow_unused=False,
                                )
    dydt = dy[:,1]
    eq = dydt + y_ini/36  # y' = -y;  36 is learnt from data

    # Boundary Condition (y(t=100) = 0) (Not used for now in final loss)
    ini_fit_ = torch.arange(5,20, dtype=torch.float32).requires_grad_(False)
    days_ = torch.ones_like(ini_fit_)*100
    x_fit_, x_days_ = torch.meshgrid(ini_fit_, days_)
    x_ss_ = torch.zeros_like(x_fit_, dtype=torch.float32)
    x_long = torch.concat([x_fit_.reshape(-1,1), x_days_.reshape(-1,1), x_ss_.reshape(-1,1)], dim=1)
    long_range_decay = torch.mean((model(x_long))**2) # y(x=100) = 0 is another boundary condition

    # Initial Condition (y(t=0) = y_prev)
    ini_fit_ = torch.arange(5,20, dtype=torch.float32).requires_grad_(False)
    days_ = torch.zeros_like(ini_fit_, dtype=torch.float32)
    x_fit_, x_days_ = torch.meshgrid(ini_fit_, days_)
    x_ss_ = torch.zeros_like(x_fit_, dtype=torch.float32)
    x_long = torch.concat([x_fit_.reshape(-1,1), x_days_.reshape(-1,1), x_ss_.reshape(-1,1)], dim=1)
    initial_condition = torch.mean((model(x_long) - x_fit_.reshape(-1,1))**2)

    # Boundary condition (y(ss=0) = y_pre)
    ini_fit_ = torch.arange(5,20, dtype=torch.float32).requires_grad_(False)
    days_ = torch.arange(1,10, dtype=torch.float32).requires_grad_(False)
    x_fit_, x_days_ = torch.meshgrid(ini_fit_, days_)
    x_ss_ = torch.zeros_like(x_fit_, dtype=torch.float32)
    x_long = torch.concat([x_fit_.reshape(-1,1), x_days_.reshape(-1,1), x_ss_.reshape(-1,1)], dim=1)
    y_out = model(x_long)
    boundary_condition = torch.mean((y_out[:,0] - y_out[:,1])**2)
    # Loss function for data
    return mse_loss(y, y_pred)*10 + torch.mean(eq**2)*10 \
        + long_range_decay*1 + initial_condition*1 + boundary_condition*1


# Define the optimization
opt = torch.optim.Adam(model.parameters(), lr=0.001)

y_dat = torch.tensor(y, dtype=torch.float32)
# Iterative learning
epochs = 10000
for epoch in range(epochs):
    opt.zero_grad()
    y_pred = model(X)
    loss = loss_func(y_dat, y_pred)

    loss.backward()  # This is where the gradient is calculated wrt the parameters and x values
    opt.step()

    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))


# Plot the results
d = np.arange(100)
ini = np.ones_like(d)*20
ss = np.ones_like(d)*0
x = np.c_[ini, d, ss]
plt.plot(d, model(x).data[:,0])
plt.plot(d, ini*np.exp(-d/36))
plt.show()


plt.scatter(y[:,0], model(X).data[:,0])
plt.scatter(y[:,1], model(X).data[:,1])
plt.plot([0,10], [0,10])
plt.show()

y_pre_pred = []
y_pred = [0.405]
for i in range(len(X)):
    X_i = X.iloc[i,:].values.reshape(1,-1)
    X_pld = X_i
    X_pld[0,0] = y_pred[i]
    y_pre_pred.append(model(X_pld).data.numpy()[0,0])
    y_pred.append(model(X_pld).data.numpy()[0,1])


t = df_train['start_timestamp'].astype("datetime64[ns]").tolist()
plt.plot(t, y_pred[1:])
plt.plot(t, y[:,1])
plt.xticks(rotation=90)
plt.show()


