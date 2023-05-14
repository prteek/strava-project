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
    ExponentialDecayEstimator,
    FitnessModel,
    PREDICTORS_FITNESS,
    expected_error,
)


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
    y = df_train[TARGET]

    logger.info("Training model")
    rest_model = ExponentialDecayEstimator()
    activity_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "estimator",
                SGDRegressor(
                    loss="squared_error",
                    penalty="l2",
                    alpha=0.01,
                    max_iter=100,
                    tol=1e-3,
                    random_state=42,
                    eta0=0.1,
                    verbose=1,
                ),
            ),
        ]
    )

    model = FitnessModel(rest_model=rest_model, activity_model=activity_model)
    model.fit(X, y)

    # Emit the required metrics
    y_true = y.values[:, 0]
    y_pred = model.predict(X)[:, 0]
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y_true, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y_true, y_pred), 3)}")

    y_true = y.values[:, 1]
    y_pred = model.predict(X)[:, 1]
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y_true, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y_true, y_pred), 3)}")

    logger.info("Saving model")
    model.PREDICTORS = PREDICTORS_FITNESS
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
