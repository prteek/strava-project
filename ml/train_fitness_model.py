#!/usr/bin/env python3
import argparse
import joblib
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from logger import logger
from helpers import (
    TARGET_FITNESS as TARGET,
    TorchModel,
    PREDICTORS_FITNESS,
    expected_error,
)
import torch


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
    y_ini = x_data[:,0]  # Value at start before decay
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
    return mse_loss(y, y_pred)*10 + torch.mean(eq**2)*5 \
        + long_range_decay*1 + initial_condition*1 + boundary_condition*1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory where to save model artefacts",
        default="/opt/ml/model",
    )

    parser.add_argument(
        "--train",
        type=str,
        help="Directory from where raw data should be read",
        default=os.environ["SM_CHANNEL_TRAIN"],  # taken care automatically in Sagemaker
    )

    args, _ = parser.parse_known_args()

    training_dir = args.train
    model_dir = args.model_dir

    logger.info("Loading training data")
    df_train = (
        pd.read_csv(os.path.join(training_dir, "train.csv"))
        .dropna(subset=[*PREDICTORS_FITNESS, *TARGET])
        .reset_index(drop=True)
    )

    X = df_train[PREDICTORS_FITNESS]
    y = df_train[TARGET].values.astype(np.float32)

    logger.info("Training model")
    torch.random.manual_seed(0)

    model = TorchModel()

    mse_loss = torch.nn.MSELoss()
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


    # Emit the required metrics
    y_true = y[:, 0]
    y_pred = model.predict(X)[:, 0]
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y_true, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y_true, y_pred), 3)}")

    y_true = y[:, 1]
    y_pred = model.predict(X)[:, 1]
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y_true, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y_true, y_pred), 3)}")

    logger.info("Saving model")
    model.PREDICTORS = PREDICTORS_FITNESS
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))