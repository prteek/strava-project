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




import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

# Define the NN model to solve the problem
class Model(nn.Module):
    def __init__(self, n_units=10, n_hidden=2):
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
                                create_graph=True, # Needed since the ODE function itself is differentiated further making this step twice differentiable
                                )

    eq = dydx + y  # y' = - 2x*y
    ic = (model(x_) - y_)**2  # y(x=0) = 1
    long_range_decay = (model(torch.tensor([[100.0]])) - 0.0)**2  # y(x=100) = 0
    return torch.mean(ic)*5 + torch.mean(eq**2)*1 + long_range_decay*0


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


