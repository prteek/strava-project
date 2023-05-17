#!/usr/bin/env python3
import argparse
import joblib
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer as FT
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from logger import logger
from helpers import (
    TARGET_FITNESS as TARGET,
    ExponentialDecayEstimator,
    FitnessModel,
    PREDICTORS_FITNESS,
    expected_error,
    dtype_converter,
)
from torch import nn
from skorch import NeuralNetRegressor


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
    estimator_ = nn.Sequential(
        nn.Linear(3, 50),
        nn.Tanh(),
        nn.Dropout(0.2),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Dropout(0.2),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 2)
    )

    estimator = NeuralNetRegressor(
        estimator_,
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

    model.fit(X, y)

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
