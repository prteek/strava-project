import awswrangler as wr
import boto3
import os
import argparse
from helpers import QUERY_ACTIVITIES, PREDICTORS, TARGET, NO_NULL_FEATURES, format_input_data
from logger import logger

boto3_session = boto3.Session(region_name="eu-west-1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where training data should be saved",
    )

    args, _ = parser.parse_known_args()
    output_dir = args.output_dir

    logger.info("Fetching training data")
    df_activities_ = (wr
                      .athena
                      .read_sql_query(QUERY_ACTIVITIES,
                                      database="strava",
                                      boto3_session=boto3_session)
                      )

    logger.info("Formatting training data")
    df_activities = (df_activities_
                     .get(PREDICTORS + [TARGET])
                     .dropna(subset=[TARGET])
                     .dropna(subset=NO_NULL_FEATURES)
                     .pipe(format_input_data)
                     )

    logger.info("Saving training data")
    df_activities.to_csv(os.path.join(output_dir, "train.csv"), index=False)

