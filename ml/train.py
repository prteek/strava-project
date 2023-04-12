from helpers import data_consistency_pipeline, PREDICTORS_, TARGET
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
from io import StringIO


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

    X = df_train[PREDICTORS_]
    y = df_train[TARGET]

    logger.info("Training model")
    model.fit(X, y)

    y_pred = model.predict(X)
    # Emit the required metrics
    rmse = round(np.sqrt(mean_squared_error(y, y_pred)))
    print(f"train_rmse: {rmse}")
    print(f"train_mean_error: {round(expected_error(y, y_pred))}")
    print(f"train_r2_score: {round(r2_score(y, y_pred), 3)}")

    logger.info("Saving model")
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
