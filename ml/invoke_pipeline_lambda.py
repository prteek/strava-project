import boto3
from datetime import datetime


client = boto3.client("sagemaker")


def invoke_training_pipeline_handler(event, context=None):
    """Lambda handler to invoke Strava training pipeline"""
    pipeline_name = "strava-ml-pipeline"
    invocation_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    response = client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName="strava-training-pipeline-execution-"
        + invocation_timestamp,
        PipelineExecutionDescription="Triggered by lambda function",
    )

    return response


def invoke_fitness_training_pipeline_handler(event, context=None):
    """Lambda handler to invoke fitness training pipeline"""
    pipeline_name = "fitness-ml-pipeline"
    invocation_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    response = client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName="fitness-training-pipeline-execution-"
        + invocation_timestamp,
        PipelineExecutionDescription="Triggered by lambda function",
    )

    return response
