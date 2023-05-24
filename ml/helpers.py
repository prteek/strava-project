import pandas as pd
from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import curve_fit
from sklearn.utils.validation import check_is_fitted
import numpy as np
import torch
from torch import nn

QUERY_ACTIVITIES = """
SELECT * FROM strava.activities
"""


feature_type_mapping = {
    "moving_time": float,
    "average_heartrate": float,
}

# These features are used by model during training and prediction
PREDICTORS = [
    "moving_time",
    "average_heartrate",
]

# Check if all required features have defined type specification
assert set(feature_type_mapping.keys()) == set(
    PREDICTORS
), "Features are missing type specification"


NO_NULL_FEATURES = [
    "moving_time",
    "average_heartrate",
]

assert set(NO_NULL_FEATURES) <= set(
    PREDICTORS
), "Columns required to be full must be a subset of input columns"


TARGET = "suffer_score"


def check_predictors(df):
    """Check if all the predictors exist in input dataframe"""
    assert set(PREDICTORS).issubset(set(df.columns)), "Predictors missing in input data"
    return df


def format_input_data(df_):
    """Format input data to pass clean version downstream
    Note: Don't try to do too much in this function"""
    df = df_.copy().astype(feature_type_mapping).pipe(check_predictors)

    return df


def raise_missing_error(df: pd.DataFrame, columns: list):
    """Raise error if missing values are found in any 'columns' input data"""
    is_nan = df.get(columns).isna().any(axis=1)
    if is_nan.any():
        raise ValueError(
            f"Missing values found in input data {df.get(columns).loc[is_nan]}"
        )
    return df


data_consistency_pipeline = Pipeline(
    [
        ("data_formatter", FT(format_input_data)),
        ("no_null_check", FT(raise_missing_error)),
    ]
)

data_consistency_pipeline.set_params(
    no_null_check__kw_args={"columns": NO_NULL_FEATURES}
)


def add_exp_heartrate(X):
    """Add exponential of heartrate (scaled for sensibility) as a new feature"""
    X = X.copy().assign(
        exp_average_heartrate=np.exp((X["average_heartrate"] - 120) / 25)
    )
    return X


def expected_error(y_true, y_pred):
    return float(np.mean(y_true - y_pred))


# ------------------ Fitness Model ------------------ #
PREDICTORS_FITNESS = ["fitness_score_initial", "time_since_last_update", "suffer_score"]
TARGET_FITNESS = ["fitness_score_pre", "fitness_score"]
PREDICTORS_ACTIVITY = ["suffer_score", "fitness_score_pre"]
PREDICTORS_REST = ["fitness_score_initial", "time_since_last_update"]


class ExponentialDecayEstimator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @staticmethod
    def decay_model(ini, t, coefs):
        """Exponential decay base model"""
        b = coefs[0]
        c = coefs[1]
        return ini * np.exp(-b * t) + c * 0

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
    def __init__(self, rest_model, activity_model):
        self.PREDICTORS_REST = PREDICTORS_REST
        self.PREDICTORS_ACTIVITY = PREDICTORS_ACTIVITY
        self.rest_model = rest_model
        self.activity_model = activity_model

    def fit(self, X, y):
        X = X.copy()
        X_rest, y_rest = X[self.PREDICTORS_REST], y["fitness_score_pre"]
        self.rest_model.fit(X_rest, y_rest)
        y_pred_rest = self.rest_model.predict(X_rest)

        X["fitness_score_pre"] = y_pred_rest
        X_activity, y_activity = X[self.PREDICTORS_ACTIVITY], y["fitness_score"]
        self.activity_model.fit(X_activity, y_activity)
        self.estimator_ = [self.rest_model, self.activity_model]

        return self

    def predict(self, X):
        X = X.copy()
        X_rest = X[self.PREDICTORS_REST]
        y_pred_rest = self.rest_model.predict(X_rest)
        X["fitness_score_pre"] = y_pred_rest
        X_activity = X[self.PREDICTORS_ACTIVITY]
        y_pred_activity = self.activity_model.predict(X_activity)
        return np.c_[y_pred_rest, y_pred_activity]


def dtype_converter(X,y=None):
    """Convert input data to float32 for torch model"""
    return X.astype(np.float32)


class TorchModel(nn.Module):
    def __init__(self, n_units=20, n_hidden=2):
        super(TorchModel, self).__init__()
        self.lin_in = nn.Linear(3,n_units)
        self.hidden = [nn.Linear(n_units,n_units) for i in range(n_hidden)]
        self.lin_out = nn.Linear(n_units,2)

    def forward(self, x):
        x = self._check_types(x)
        x = torch.relu(self.lin_in(x))
        for i_hidden in self.hidden:
            x = torch.relu(i_hidden(x))

        x = self.lin_out(x)
        return x

    @staticmethod
    def _check_types(x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise TypeError(f"Input type {type(x)} not supported")
        return x

    def predict(self, x):
        x = self._check_types(x)
        return self.forward(x).detach().numpy()
