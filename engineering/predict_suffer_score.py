import json
import os
import time
import pandas as pd
import awswrangler as wr
import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer, JSONSerializer


def handler(event, context=None):
    # TODO implement
    print(event)
    PREDICTORS_ = [
        "moving_time",
        "average_heartrate",
    ]
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


if __name__ == '__main__':
    handler({'test': 'test'})