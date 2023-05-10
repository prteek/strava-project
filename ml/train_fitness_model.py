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
from helpers import (TARGET_FITNESS as TARGET,
                    ExponentialDecayEstimator,
                    FitnessModel,
                    PREDICTORS_ACTIVITY,
                    PREDICTORS_REST,
                    PREDICTORS_FITNESS,
                    expected_error
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
    df_train = (pd
                .read_csv(os.path.join(training_dir, "train.csv"))
                .astype({"fitness_score": float, "suffer_score": float, "date": "datetime64"})
                .sort_values("date")
                )

    logger.info("Training rest model")
    # We want to predict future fitness based on current fitness and time of inactivity in future

    df_rest = (df_train
               .get(['fitness_score','id', 'date'])
               .assign(time_since_last_update=lambda x: (x['date'].shift(-1) - x['date']).dt.days,
                       fitness_score_final=lambda x: x['fitness_score'].shift(-1),
                       activity_in_between=lambda x: np.logical_not(x['id'].shift(-1).isna()))
               .query("activity_in_between == False")
               .rename({"fitness_score": "fitness_score_initial",
                        "fitness_score_final": "fitness_score"}, axis=1)
               .get([TARGET, *PREDICTORS_REST])
               .dropna(subset=[TARGET, *PREDICTORS_REST])
               )

    X = df_rest[PREDICTORS_REST]
    y = df_rest[TARGET]

    rest_model = ExponentialDecayEstimator()

    rest_model.fit(X, y)
    rest_model.PREDICTORS = PREDICTORS_REST

    y_pred = rest_model.predict(X)
    # Emit the required metrics
    rmse = round(np.sqrt(mean_squared_error(y, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y, y_pred), 3)}")

    logger.info("Training activity model")
    # We want to predict current fitness based on past fitness and current suffer score
    df_activity = (df_train
                   .assign(fitness_score_initial=lambda x: x['fitness_score'].shift(1))
                   .get([TARGET, *PREDICTORS_ACTIVITY])
                   .dropna(subset=[TARGET, *PREDICTORS_ACTIVITY])
                   )

    X = df_activity[PREDICTORS_ACTIVITY]
    y = df_activity[TARGET]

    activity_model = Pipeline([('scaler', StandardScaler()),
                               ('estimator', SGDRegressor(loss='squared_error',
                                                          penalty='l2',
                                                          alpha=0.01,
                                                          max_iter=100,
                                                          tol=1e-3,
                                                          random_state=42,
                                                          eta0=0.1,
                                                          verbose=1))
                               ])

    activity_model.fit(df_activity[PREDICTORS_ACTIVITY], df_activity[TARGET])
    activity_model.PREDICTORS = PREDICTORS_ACTIVITY

    y_pred = activity_model.predict(X)
    rmse = round(np.sqrt(mean_squared_error(y, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y, y_pred), 3)}")

    logger.info("Saving model")
    model = FitnessModel(activity_model, rest_model)
    model.PREDICTORS = PREDICTORS_FITNESS
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
