#!/usr/bin/env python3
import sys
import awswrangler as wr
import boto3
import os
import argparse
import pandas as pd
from glob import glob

# Imports from local files
sys.path.append(
    "/opt/ml/processing/input"
)  # To be able to read local files in container

from logger import logger


boto3_session = boto3.Session(region_name="eu-west-1")


def read_and_parse_fitness_data(file):
    df_r = (
        pd.read_csv(file, parse_dates=["x"], dayfirst=True)
        .rename({" y": "fitness_score", "x": "timestamp"}, axis=1)
        .astype({"fitness_score": float, "timestamp": "datetime64[ns]"})
        .assign(
            date=lambda x: x["timestamp"].dt.date.astype("str"),
            source=lambda x: file.split("/")[-1],
        )
        .sort_values("date")
    )
    return df_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where training data should be saved",
        default="/opt/ml/processing/output",
    )

    parser.add_argument(
        "--fitness-data-dir",
        type=str,
        help="Directory where fitness data is stored",
        default="/opt/ml/processing/input/fitness_data",
    )

    args, _ = parser.parse_known_args()
    output_dir = args.output_dir
    fitness_data_dir = args.fitness_data_dir

    logger.info("reading fitness data data")
    file_list = glob(os.path.join(fitness_data_dir, "*.csv"))
    df_fitness = (
        pd.concat([read_and_parse_fitness_data(file) for file in file_list])
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    start_date = df_fitness["date"].min()
    end_date = df_fitness["date"].max()

    logger.info(f"reading activities data from {start_date} to {end_date}")
    df_activities = (
        wr.athena.read_sql_query(
            f"""SELECT * from strava.activities
                                    where start_timestamp >= date('{start_date}')
                                    and start_timestamp <= date('{end_date}')
                                    """,
            "strava",
            boto3_session=boto3_session,
        )
        .assign(
            date=lambda x: x["start_timestamp"].dt.date.astype("str")
        )  # Take date for the activity
        .sort_values("date")
    )

    df_merged = (
        pd.merge(df_fitness, df_activities, on="date", how="outer")
        .sort_values("timestamp")
        .assign(fitness_score_pre=lambda x: x["fitness_score"].shift(1))
        .dropna(subset=["id"])
        .assign(
            time_since_last_update=lambda x: (
                x["date"].astype("datetime64[ns]").diff()
            ).dt.days,
            fitness_score_initial=lambda x: x["fitness_score"].shift(1),
        )
        .astype(
            {
                "fitness_score": float,
                "date": "datetime64[ns]",
                "id": float,
                "suffer_score": float,
            }
        )
    )

    logger.info(f"writing training data to {output_dir}")
    df_merged.to_csv(os.path.join(output_dir, "train.csv"), index=False)

# %%
