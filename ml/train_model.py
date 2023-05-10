#!/usr/bin/env python3
from helpers import data_consistency_pipeline, PREDICTORS, TARGET, expected_error
import argparse
import joblib
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from logger import logger


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
    df_train = pd.read_csv(os.path.join(training_dir, "train.csv"))

    preprocessing = Pipeline(
        [
            ("data_consistency", data_consistency_pipeline),
            ("scaler", StandardScaler()),
            (
                "feature_engineering",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
        ]
    )

    estimator = PoissonRegressor(alpha=1)
    model = Pipeline([("preprocessing", preprocessing), ("estimator", estimator)])

    # model.set_output(transform="pandas")
    # model.set_params(feature_engineering=None)

    X = df_train[PREDICTORS]
    y = df_train[TARGET]

    logger.info("Training model")
    model.fit(X, y)
    model.PREDICTORS = PREDICTORS

    y_pred = model.predict(X)
    # Emit the required metrics
    rmse = round(np.sqrt(mean_squared_error(y, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y, y_pred), 3)}")

    logger.info("Saving model")
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
