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

X = df_train[PREDICTORS_FITNESS]/np.array([1,1,10])
y = df_train[TARGET].values.astype(np.float32)


class Model(nn.Module):
    def __init__(self, n_units=20, n_hidden=2):
        super(Model, self).__init__()
        self.lin_in = nn.Linear(3,n_units)
        self.hidden = [nn.Linear(n_units,n_units) for i in range(n_hidden)]
        self.lin_out = nn.Linear(n_units,2)

    def forward(self, x):
        x = torch.relu(self.lin_in(x))
        for i_hidden in self.hidden:
            x = torch.relu(i_hidden(x))

        x = self.lin_out(x)
        return x



model = Model()

mse_loss = torch.nn.MSELoss()

def ODE(X_,y_):
    # There is a small precaution in using the gradient with vector input.
    # You should feed the backward function with unit vector in order to access the gradient as a vector.
    y_fit_pre = y_[:,0]
    y_ini = x_fit
    dydt, = torch.autograd.grad(y_fit_pre, x_days,
                                grad_outputs=torch.ones_like(y_fit_pre),
                                create_graph=True,  # Needed since the ODE function itself is differentiated further
                                # in training loop making this step twice differentiable,
                                allow_unused=True,
                                )

    eq = dydt + y_ini/36  # y' = -y

    ini_fit_ = torch.arange(5,10, dtype=torch.float32).requires_grad_(False)
    days_ = torch.ones_like(ini_fit_)*100
    x_fit_, x_days_ = torch.meshgrid(ini_fit_, days_)
    x_ss_ = torch.zeros_like(x_fit_, dtype=torch.float32)/10.0
    x_long = torch.concat([x_fit_.reshape(-1,1), x_days_.reshape(-1,1), x_ss_.reshape(-1,1)], dim=1)
    long_range_decay = torch.mean((model(torch.tensor(x_long)))**2) # y(x=10) = 0 is another boundary condition

    ini_fit_ = torch.arange(5,20, dtype=torch.float32).requires_grad_(False)
    days_ = torch.zeros_like(ini_fit_, dtype=torch.float32)
    x_fit_, x_days_ = torch.meshgrid(ini_fit_, days_)
    x_ss_ = torch.zeros_like(x_fit_, dtype=torch.float32)/10.0
    x_long = torch.concat([x_fit_.reshape(-1,1), x_days_.reshape(-1,1), x_ss_.reshape(-1,1)], dim=1)

    initial_condition = torch.mean((model(torch.tensor(x_long)) - x_fit_.reshape(-1,1))**2)

    X_dat = torch.tensor(X.values, dtype=torch.float32)
    y_dat = torch.tensor(y, dtype=torch.float32)
    return mse_loss(y_dat, model(X_dat)) + torch.mean(eq**2)*10 \
        + long_range_decay*0 + initial_condition*1


loss_func = ODE


# Define the optimization
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Define reference grid
days = torch.arange(1,10, dtype=torch.float32).requires_grad_(True)
ini_fit = torch.arange(5,20, dtype=torch.float32).requires_grad_(True)
x_fit, x_days = torch.meshgrid(ini_fit, days)
x_ss = torch.zeros_like(x_fit, dtype=torch.float32)/10.0
x_data = torch.concat([x_fit.reshape(-1,1), x_days.reshape(-1,1), x_ss.reshape(-1,1)], dim=1)
# Iterative learning
epochs = 5000
for epoch in range(epochs):
    torch.random.manual_seed(0)
    opt.zero_grad()
    y_trial = model(x_data)
    loss = loss_func(x_data, y_trial)

    loss.backward()  # This is where the gradient is calculated wrt the parameters and x values
    opt.step()

    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))



d = np.arange(10)
ini = np.ones_like(d)*20
ss = np.ones_like(d)*0
x = torch.tensor(np.c_[ini, d, ss], dtype=torch.float32)/torch.tensor([1,1,10], dtype=torch.float32)
plt.plot(d, model(x).data[:,0])
plt.plot(d, ini*np.exp(-0.028*d))
# plt.scatter([0,35], [6.2,2.3])
plt.show()

plt.scatter(y[:,0], model(torch.tensor(X.values, dtype=torch.float32)).data[:,0])
plt.scatter(y[:,1], model(torch.tensor(X.values, dtype=torch.float32)).data[:,1])
plt.plot([0,10], [0,10])
plt.show()



import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

# Define the NN model to solve the problem
class Model(nn.Module):
    def __init__(self, n_units=20, n_hidden=2):
        super(Model, self).__init__()
        self.lin_in = nn.Linear(1,n_units)
        self.hidden = [nn.Linear(n_units,n_units) for i in range(n_hidden)]
        self.lin_out = nn.Linear(n_units,1)

    def forward(self, x):
        x = torch.relu(self.lin_in(x))
        for i_hidden in self.hidden:
            x = torch.relu(i_hidden(x))

        x = self.lin_out(x)
        return x



model = Model()

torch.random.manual_seed(0)
x_ = 2*(torch.rand(size=(10,1))-0.5)
noise = (torch.rand(size=(10,1)) - 0.5)*0.05
y_ = torch.exp(- x_) + noise

# plt.scatter(x_.detach().numpy(), y_.detach().numpy())
# plt.show()
# Define loss_function from the Ordinary differential equation to solve


def ODE(x,y):
    # There is a small precaution in using the gradient with vector input.
    # You should feed the backward function with unit vector in order to access the gradient as a vector.
    dydx, = torch.autograd.grad(y, x,
                                grad_outputs=torch.ones_like(y),
                                create_graph=True,  # Needed since the ODE function itself is differentiated further
                                # in training loop making this step twice differentiable
                                )

    eq = dydx + y  # y' = -y
    ic = (y_ - model(x_))**2  # y = y_true at x = x_true is boundary condition of sorts
    # long_range_decay = (model(torch.tensor([[10.0]])) - 0.0)**2  # y(x=10) = 0 is another boundary condition
    return torch.mean(ic)*5 + torch.mean(eq**2)*1


loss_func = ODE

# Define the optimization
opt = optim.Adam(model.parameters(), lr=0.005)

# Define reference grid
x_data_ = torch.linspace(-2,10,100).view(-1,1)
x_data = x_data_.data.new(x_data_).requires_grad_(True)
# Iterative learning
epochs = 5000
for epoch in range(epochs):
    torch.random.manual_seed(0)
    opt.zero_grad()
    y_trial = model(x_data)
    loss = loss_func(x_data, y_trial)

    loss.backward()  # This is where the gradient is calculated wrt the parameters and x values
    opt.step()

    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))


y_data = model(x_data)
x = torch.linspace(-2,10,1000).view(-1,1)
y = model(x)
# Plot Results
plt.scatter(x_.data.numpy(), y_.data.numpy(), label='data')
plt.plot(x, np.exp(-x), label='exact')
plt.plot(x, y.data, label='approx')
plt.fill_between([x_.data.min(), x_.data.max()], 0,5, label='training range', alpha=0.2)
plt.legend()
plt.show()



#%%
import torch
from torch import nn




