#!/usr/bin/env python
import joblib
import os
import pandas as pd
import numpy as np
from io import StringIO
import flask
from flask import Flask, Response

# Put the following files in /usr/bin in Docker container
from helpers import TorchModel
from logger import logger


def model_fn(model_dir):
    """Load the model from the `model_dir` directory."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(input_data, content_type):
    """Parse input data payload as properly formatted dataframe"""
    try:
        if content_type == "text/csv":
            # Read the raw input data as CSV.
            input_data_ = input_data.decode("utf-8")
            X = pd.read_csv(StringIO(input_data_), header=None)
            return X

    except Exception as e:
        logger.error(e)
        logger.error(f"Error parsing input data {input_data}")
        return None


def predict_fn(input_data: pd.DataFrame, model):
    """Predict using the model and input data"""
    try:
        input_data.columns = model.PREDICTORS
        predictions_raw = model.predict(input_data)
        predictions = [
            list(value) if isinstance(value, np.ndarray) else value
            for value in predictions_raw
        ]
        return predictions
    except Exception as e:
        logger.error(e)
        logger.error(f"Error predicting on input data {input_data}")
        return None


app = Flask(__name__)
model = model_fn(model_dir="/opt/ml/model")


@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="\n", status=200)


@app.route("/invocations", methods=["POST"])
def predict():
    """Compound prediction function for the model"""

    # Read the input data into pandas dataframe
    input_data = flask.request.data
    content_type = flask.request.content_type
    X = input_fn(input_data, content_type)

    if X is None:
        return Response(
            response="Parsing process failure", status=400, mimetype="text/plain"
        )

    else:
        # Predict on the data
        predictions = predict_fn(X, model)
        if predictions is None:
            return Response(
                response="Prediction process failure", status=400, mimetype="text/plain"
            )
        else:
            response = str(predictions)
            return Response(response=response, status=200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)  # Same port as in Dockerfile
