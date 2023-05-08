#!/usr/bin/env python3
from helpers import data_consistency_pipeline, PREDICTORS_, TARGET
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
from io import StringIO
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import curve_fit
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


def model_fn(model_dir):
    """Load the model from the `model_dir` directory."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(input_data: str, content_type):
    """Parse input data payload as properly formatted dataframe"""
    try:
        if content_type == "text/csv":
            # Read the raw input data as CSV.
            X = pd.read_csv(StringIO(input_data), names=PREDICTORS_)
            return X

    except Exception as e:
        logger.error(e)
        logger.error(f"Error parsing input data {input_data}")


def predict_fn(input_data: pd.DataFrame, model):
    """Predict using the model and input data"""
    try:
        predictions_raw = model.predict(input_data)
        predictions = [round(value) for value in predictions_raw]
        return predictions
    except Exception as e:
        logger.error(e)
        logger.error(f"Error predicting on input data {input_data}")


def expected_error(y_true, y_pred):
    return float(np.mean(y_true - y_pred))


class ExponentialDecayEstimator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @staticmethod
    def decay_model(ini, t, coefs):
        """Exponential decay base model"""
        b = coefs[0]
        c = coefs[1]
        return ini * np.exp(-b * t) + c

    @staticmethod
    def _exponential_decay_model_opt(x, b, c):
        """Exponential decay model function to fit"""
        ini = x[:, 0]
        t = x[:, 1]
        return ExponentialDecayEstimator.decay_model(ini, t, [b, c])

    def _format_input(self, X):
        if isinstance(X, pd.DataFrame):
            X_ = X.values
        else:
            X_ = X

        return X_

    def fit(self, X, y):
        """Fit the model to the data"""
        X_ = self._format_input(X)
        self.coef_, _ = curve_fit(self._exponential_decay_model_opt, X_, y)
        return self

    def predict(self, X):
        """Predict using the model"""
        X_ = self._format_input(X)
        return self._exponential_decay_model_opt(X_, *self.coef_)


class FitnessModel(BaseEstimator, TransformerMixin):
    def __init__(self, activity_model, rest_model):
        self.activity_model = activity_model
        self.rest_model = rest_model
        check_is_fitted(self.activity_model, msg="Activity model not fitted")
        check_is_fitted(self.rest_model, msg="Rest model not fitted")

    def predict(self, X):
        """Predict using the model"""
        X_ = X.copy()
        X_['fitness_score'] = -99999.0
        i_activity = X_['suffer_score'] > 0
        PREDICTORS_ACTIVITY = self.activity_model.PREDICTORS
        X_.loc[i_activity, 'fitness_score'] = self.activity_model.predict(X_.loc[i_activity, PREDICTORS_ACTIVITY])

        i_rest = X_['suffer_score'] == 0
        PREDICTORS_REST = self.rest_model.PREDICTORS
        X_.loc[i_rest, 'fitness_score'] = self.rest_model.predict(X_.loc[i_rest, PREDICTORS_REST])

        return X_['fitness_score'].values


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

rest_model = ExponentialDecayEstimator()

PREDICTORS = ["suffer_score", "fitness_score_initial", "time_since_last_update"]
PREDICTORS_ACTIVITY = ["suffer_score", "fitness_score_initial"]
PREDICTORS_REST = ["fitness_score_initial", "time_since_last_update"]

TARGET = "fitness_score"

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

    activity_model.fit(df_activity[PREDICTORS_ACTIVITY], df_activity[TARGET])
    activity_model.PREDICTORS = PREDICTORS_ACTIVITY

    y_pred = activity_model.predict(X)
    rmse = round(np.sqrt(mean_squared_error(y, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y, y_pred), 3)}")

    logger.info("Saving model")
    model = FitnessModel(activity_model, rest_model)

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
