import os
import sys
import boto3
import argparse
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.model import Model
import time
import configparser
from datetime import datetime


config = configparser.ConfigParser()
config.read("config.txt")
region = os.environ["AWS_DEFAULT_REGION"]
role = os.environ["SAGEMAKER_EXECUTION_ROLE"]

boto3_session = boto3.Session(region_name=region)
session = sagemaker.Session()
sm_client = boto3.client("sagemaker", region_name=region)


def upload_code_helpers(filepath_list: list, bucket: str, prefix: str) -> str:
    for filepath in filepath_list:
        _ = session.upload_data(filepath, bucket, key_prefix=prefix)

    return f"s3://{bucket}/{prefix}/"


class ModelAdapter:
    """
    Adapter to add update_serverless_endpoint method to sagemaker Model class
    """

    def __init__(self, model: Model, sagemaker_client):
        self.model = model
        self.sm_client = sagemaker_client

    def _create_serverless_epc(
        self, endpoint_config_name: str, memory_size_in_mb: int, max_concurrency: int
    ):
        """Create a new End point config for serverless inference,
         that uses current model. This config is created for each deployment update as
        lineage tracking for endpoints"""

        create_endpoint_config_response = self.sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "ModelName": self.model.name,
                    "VariantName": "AllTraffic",
                    "ServerlessConfig": {
                        "MemorySizeInMB": memory_size_in_mb,
                        "MaxConcurrency": max_concurrency,
                    },
                }
            ],
        )

        return create_endpoint_config_response

    def update_serverless_endpoint(
        self,
        endpoint_name: str,
        endpoint_config_base_name: str,
        memory_size_in_mb: int = 1024,
        max_concurrency: int = 5,
    ):
        """Update existing end point with a new model"""
        endpoint_config_name = f"{endpoint_config_base_name}-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
        _ = self._create_serverless_epc(
            endpoint_config_name, memory_size_in_mb, max_concurrency
        )

        _ = self.sm_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )

        # wait for endpoint to reach a terminal state (InService) using describe endpoint
        describe_endpoint_response = self.sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )

        while describe_endpoint_response["EndpointStatus"] == "Updating":
            describe_endpoint_response = self.sm_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            print(
                f"{describe_endpoint_response['EndpointStatus']} Endpoint: {endpoint_name}"
            )
            time.sleep(10)

        return describe_endpoint_response

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object"""
        return getattr(self.model, attr)


def check_endpoint_exists(endpoint_name: str) -> bool:
    """Check if an endpoint already exists"""
    response_blob = sm_client.list_endpoints()
    endpoint_names = [e["EndpointName"] for e in response_blob["Endpoints"]]
    return endpoint_name in endpoint_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--on-aws",
        action="store_true",
        help="If the orchestration needs to be done on aws set this flag to true",
        default=False,
    )

    args, _ = parser.parse_known_args()
    on_aws = args.on_aws

    if on_aws:
        processor_instance_type = "ml.m5.large"
        train_instance_type = "ml.m5.large"
    else:
        processor_instance_type = "local"
        train_instance_type = "local"

    bucket = config.get("aws", "bucket")
    image_uri = os.environ["IMAGE_URI"]
    processing_local_dependencies = ["logger.py", "helpers.py"]
    helpers = upload_code_helpers(
        processing_local_dependencies, bucket, prefix="helpers"
    )

    # ----- Fetch training data ----- #
    fetch_data_code_location = session.upload_data(
        "prepare_training_data.py", bucket=bucket, key_prefix="fetch-data/code"
    )

    fetch_data_output = f"s3://{bucket}/fetch-data/output/"

    fetch_data = ScriptProcessor(
        base_job_name="fetch-data",
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_count=1,
        instance_type=processor_instance_type,
    )

    fetch_data.run(
        fetch_data_code_location,
        inputs=[ProcessingInput(helpers, "/opt/ml/processing/input")],
        outputs=[ProcessingOutput("/opt/ml/processing/output/", fetch_data_output)],
    )

    # ----- Train model ----- #
    train_data_location = fetch_data.latest_job.outputs[0].destination
    train_output_location = (
        f"s3://{bucket}/train/job-artefacts"  # Model artefacts will be uploaded here
    )
    local_dependencies = ["logger.py", "helpers.py"]

    estimator = Estimator(
        base_job_name="strava-training",
        role=role,
        entry_point="train.py",
        image_uri=image_uri,
        instance_count=1,
        instance_type=train_instance_type,
        dependencies=local_dependencies,
        code_location=train_output_location,
        output_path=train_output_location,
    )

    estimator.fit({"train": train_data_location})

    if not on_aws:  # Halt script execution here if this is a test
        print("Local orchestration test complete")
        sys.exit(0)
    else:
        # ----- Deploy model ----- #
        model_name = config.get("model", "name")
        endpoint_config_name = config.get("endpoint", "config-name")
        endpoint_name = config.get("endpoint", "name")
        code_location = f"s3://{bucket}/train/Model"  # Code files will be uploaded here

        _model = Model(
            image_uri=estimator.image_uri,
            model_data=estimator.model_data,
            sagemaker_session=session,
            role=role,
            name=model_name,
            entry_point="train.py",
            code_location=code_location,
            dependencies=local_dependencies,
        )

        model = ModelAdapter(_model, sm_client)
        model.create()

        memory_size_in_mb = int(config.get("endpoint", "memory-size-in-mb"))
        max_concurrency = int(config.get("endpoint", "max-concurrency"))

        if check_endpoint_exists(endpoint_name):
            model.update_serverless_endpoint(
                endpoint_name=endpoint_name,
                endpoint_config_base_name=endpoint_config_name,
                memory_size_in_mb=memory_size_in_mb,
                max_concurrency=max_concurrency,
            )

        else:  # Create a new endpoint
            sic = ServerlessInferenceConfig(
                memory_size_in_mb=memory_size_in_mb, max_concurrency=max_concurrency
            )
            model.deploy(endpoint_name=endpoint_name, serverless_inference_config=sic)
