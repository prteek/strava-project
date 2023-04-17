import json
import pandas as pd
import awswrangler as wr
import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer, JSONSerializer
from datetime import datetime

PREDICTORS_ = [
    "moving_time",
    "average_heartrate",
]  # Check with ML model implementation or create a dependency

QUERY_ACTIVITY_IDS = """
SELECT *
FROM strava.activities
WHERE id IN {activity_ids}
"""

boto3_session = boto3.Session(region_name="eu-west-1")


def handler(event, context=None):
    predictor = Predictor("strava", serializer=CSVSerializer())
    activity_ids = tuple(json.loads(event["body"]["ActivityIds"]))
    if len(activity_ids) == 0:
        return {"statusCode": 200, "body": json.dumps("No activities to predict on")}

    else:
        df_activities = wr.athena.read_sql_query(
            QUERY_ACTIVITY_IDS.format(activity_ids=activity_ids),
            database="strava",
            boto3_session=boto3_session,
        )

        X = df_activities[PREDICTORS_].values
        y = eval(predictor.predict(X).decode())

        results_dump = []
        for i, activity_id in enumerate(activity_ids):
            i_result = {
                "activity_id": activity_id,
                "predicted_suffer_score": y[i],
                "prediction_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            }
            results_dump.append(i_result)

        df_results = pd.DataFrame(results_dump).astype(
            {"prediction_timestamp": "datetime64[s]"}
        )

        wr.s3.to_csv(
            df=df_results,
            path="s3://pp-strava-data/activities/predicted_suffer_score",
            index=False,
            dataset=True,
            mode="append",
            database="strava",
            table="predicted_suffer_score",
            boto3_session=boto3_session,
        )

        return {"statusCode": 200, "body": json.dumps(results_dump)}
