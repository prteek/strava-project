import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.pipeline import Pipeline


QUERY_ACTIVITIES = """
SELECT * FROM strava.activities
"""


feature_type_mapping = {
    "moving_time": float,
    "average_heartrate": float,
}

# These features are used by model during training and prediction
PREDICTORS_ = [
    "moving_time",
    "average_heartrate",
]

# Check if all required features have defined type specification
assert set(feature_type_mapping.keys()) == set(
    PREDICTORS_
), "Features are missing type specification"


# We want more inputs like id compared to what model uses to make predictions for debugging etc.
PREDICTORS = [
    "id",
] + PREDICTORS_

# What model uses to make predictions must be a subset of what we have as inputs
assert set(PREDICTORS_) <= set(
    PREDICTORS
), "Columns defined to be used by model are not a subset of input columns"

NO_NULL_FEATURES = [
    "moving_time",
    "average_heartrate",
]

assert set(NO_NULL_FEATURES) <= set(
    PREDICTORS_
), "Columns required to be full must be a subset of input columns"


TARGET = "suffer_score"


def check_predictors(df):
    """Check if all the predictors exist in input dataframe"""
    assert set(PREDICTORS_).issubset(
        set(df.columns)
    ), "Predictors missing in input data"
    return df


def format_input_data(df_):
    """Format input data to pass clean version downstream
    Note: Don't try to do too much in this function"""
    df = (df_.copy()
          .astype(feature_type_mapping)
          .pipe(check_predictors)
          )

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
