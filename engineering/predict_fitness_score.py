import json
import pandas as pd
import numpy as np
import awswrangler as wr
import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from datetime import datetime

PREDICTORS_FITNESS = ["fitness_score_initial", "time_since_last_update", "suffer_score"]

QUERY_ACTIVITY_TIMESTAMP = """
SELECT start_timestamp 
FROM strava.activities
WHERE id = {activity_id}
"""

QUERY_PREVIOUS_FITNESS = f"""
SELECT fitness_score,
       start_timestamp
FROM strava.predicted_fitness_score
WHERE start_timestamp < ({QUERY_ACTIVITY_TIMESTAMP})
                         
ORDER BY start_timestamp DESC
LIMIT 1
"""


boto3_session = boto3.Session(region_name="eu-west-1")


def add_timestamps(blob):
    """Add start_timestamp to each activity in blob"""
    for i in blob:
        activity_id = i["activity_id"]
        t = wr.athena.read_sql_query(QUERY_ACTIVITY_TIMESTAMP.format(activity_id=activity_id),
                                     "strava",
                                     boto3_session=boto3_session,
                                     ctas_approach=False)
        i.update({"start_timestamp": t["start_timestamp"].values[0]})
    return None


def handler(event, context=None):
    predictor = Predictor("strava-fitness", serializer=CSVSerializer())
    suffer_score_blob = json.loads(event["body"])

    if len(suffer_score_blob) == 0:
        return {"statusCode": 200, "body": json.dumps([])}

    else:
        result_blob = []
        add_timestamps(suffer_score_blob)
        suffer_score_blob_sorted = sorted(suffer_score_blob, key=lambda x: x["start_timestamp"])
        for activity in suffer_score_blob_sorted:
            print(f"Processing activity {activity['activity_id']}")
            activity_id = activity['activity_id']
            activity_timestamp = activity['start_timestamp']

            previous_activity = wr.athena.read_sql_query(
                QUERY_PREVIOUS_FITNESS.format(activity_id=activity_id),
                "strava",
                boto3_session=boto3_session,
                ctas_approach=False).to_dict(orient="records")[0]

            time_since_last_update = np.timedelta64(activity_timestamp - previous_activity["start_timestamp"], 'D').astype(int)
            fitness_score_initial = previous_activity["fitness_score"]
            suffer_score = activity["predicted_suffer_score"]
            payload = [[fitness_score_initial, time_since_last_update, suffer_score]]
            predictions = eval(predictor.predict(payload).decode())[0]
            activity_fitness = {"activity_id": activity_id,
                                "start_timestamp": activity_timestamp,
                                "fitness_score_pre": round(predictions[0],2),
                                "fitness_score": round(predictions[1],2),
                                "prediction_timestamp": datetime.now()}

            df_activity_fitness = (pd
                                   .DataFrame([activity_fitness])
                                   .astype({"activity_id": float,
                                            "start_timestamp": "datetime64[s]",
                                            "fitness_score_pre": float,
                                            "fitness_score": float,
                                            "prediction_timestamp": "datetime64[s]"})
                                   )

            wr.s3.to_csv(
                df=df_activity_fitness,
                path="s3://pp-strava-data/activities/predicted_fitness_score",
                index=False,
                dataset=True,
                mode="overwrite_partitions",
                partition_cols=["activity_id"],
                database="strava",
                table="predicted_fitness_score",
                boto3_session=boto3_session,
            )

            result_blob.append({"activity_id": activity_id,
                                "fitness_score_pre": predictions[0],
                                "fitness_score": predictions[1]})

        return {"statusCode": 200, "body": json.dumps(result_blob)}

