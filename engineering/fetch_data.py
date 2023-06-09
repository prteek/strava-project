import json
import os
import time
import pandas as pd
from stravalib.client import Client
from clumper import Clumper
import awswrangler as wr
import boto3
from datetime import datetime, timedelta


STRAVA_CLIENT_ID = os.environ["STRAVA_CLIENT_ID"]
STRAVA_CLIENT_SECRET = os.environ["STRAVA_CLIENT_SECRET"]
STRAVA_ACCESS_TOKEN = os.environ["STRAVA_ACCESS_TOKEN"]
STRAVA_REFRESH_TOKEN = os.environ["STRAVA_REFRESH_TOKEN"]
EXPIRES_AT = os.environ["EXPIRES_AT"]

access_token = dict(
    access_token=STRAVA_ACCESS_TOKEN,
    refresh_token=STRAVA_REFRESH_TOKEN,
    expires_at=float(EXPIRES_AT),
)


def stream_to_df(stream):
    """Converts a stream to a dataframe"""
    df_stream = pd.DataFrame()
    for k, v_ in stream.items():
        v = v_.dict()["data"]
        if k == "latlng":
            df_stream["lat"] = [x[0] for x in v]
            df_stream["lng"] = [x[1] for x in v]
        else:
            df_stream[k] = v

    return df_stream


keep_activity_cols = [
    "id",
    "name",
    "distance",
    "moving_time",
    "elapsed_time",
    "type",
    "start_date",
    "location_country",
    "average_speed",
    "average_watts",
    "suffer_score",
    "has_heartrate",
    "average_cadence",
    "average_heartrate",
    "max_heartrate",
]

channels = [
    "time",
    "distance",
    "latlng",
    "altitude",
    "velocity_smooth",
    "moving",
    "grade_smooth",
    "heartrate",
    "cadence",
    "watts",
]

stream_columns_to_save = ["moving",
                           "grade_smooth",
                           "heartrate",
                           "time",
                           "velocity_smooth",
                           "watts",
                           "lat",
                           "lng",
                           "cadence",
                           "distance",
                           "altitude",
                           "activity_id"]


strava_client = Client()
boto3_session = boto3.Session(region_name="eu-west-1")


def handler(event, context=None):
    if time.time() > access_token["expires_at"]:
        print("Token has expired, will refresh")
        refresh_response = strava_client.refresh_access_token(
            client_id=STRAVA_CLIENT_ID,
            client_secret=STRAVA_CLIENT_SECRET,
            refresh_token=access_token["refresh_token"],
        )

        strava_client.access_token = refresh_response["access_token"]
        strava_client.refresh_token = refresh_response["refresh_token"]
        strava_client.token_expires_at = refresh_response["expires_at"]

    else:
        print(
            "Token still valid, expires at {}".format(
                time.strftime(
                    "%a, %d %b %Y %H:%M:%S %Z",
                    time.localtime(access_token["expires_at"]),
                )
            )
        )
        strava_client.access_token = access_token["access_token"]
        strava_client.refresh_token = access_token["refresh_token"]
        strava_client.token_expires_at = access_token["expires_at"]

    athlete = strava_client.get_athlete().to_dict()
    if (event is not None) and ('start_date' in event):
        start_date = datetime.strptime(event['start_date'], '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=1)

    activities_response = strava_client.get_activities(
        after=start_date.strftime("%Y-%m-%d")
    )
    activities = Clumper([a.dict() for a in activities_response])

    if activities.shape[0] == 0:
        print("No activities found")
        return {"statusCode": 200, "body": json.dumps([])}
    else:
        activities = activities.select(*keep_activity_cols)
        streams = []
        for id_ in activities.select("id").collect():
            activity_stream = strava_client.get_activity_streams(
                id_["id"], types=channels
            )
            if activity_stream is not None:
                df_stream = stream_to_df(activity_stream).assign(activity_id=id_["id"])
                streams.append(df_stream)

        # Create an empty dataframe with required columns then append stream data to it
        # This ensures that irrespective of Workout type all required columns exist
        # to be able to add data to table safely
        df_streams = (pd
                      .concat([pd.DataFrame(columns=stream_columns_to_save),
                               pd.concat(streams)]
                              )
                      )

        df_activities = (pd
                        .DataFrame(activities.collect())
                        .rename(columns={"start_date": "start_timestamp"})
                        .assign(start_date=lambda x: x["start_timestamp"].dt.date)
                        )

        wr.s3.to_csv(
            df=df_activities,
            path="s3://pp-strava-data/activities/metadata",
            index=False,
            dataset=True,
            mode="overwrite_partitions",
            partition_cols=["start_date"],
            database="strava",
            table="activities",
            boto3_session=boto3_session,
        )

        wr.s3.to_csv(
            df=df_streams,
            path="s3://pp-strava-data/activities/streams",
            index=False,
            dataset=True,
            mode="overwrite_partitions",
            partition_cols=["activity_id"],
            database="strava",
            table="streams",
            boto3_session=boto3_session,
        )

        activity_ids = [{"activity_id": i} for i in df_activities["id"].tolist()]

        return {
            "statusCode": 200,
            "body": json.dumps(activity_ids),
        }
