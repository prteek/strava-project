import json
import boto3
import os
from sagemaker.model import Model
from datetime import datetime
from sagemaker.serverless import ServerlessInferenceConfig
import time

role = os.environ["SAGEMAKER_EXECUTION_ROLE"]
region = os.environ["AWS_DEFAULT_REGION"]
sm_client = boto3.client("sagemaker", region_name=region)


def check_endpoint_exists(endpoint_name: str) -> bool:
    """Check if an endpoint already exists"""
    response_blob = sm_client.list_endpoints()
    endpoint_names = [e["EndpointName"] for e in response_blob["Endpoints"]]
    return endpoint_name in endpoint_names


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

        # # wait for endpoint to reach a terminal state (InService) using describe endpoint
        describe_endpoint_response = self.sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        #
        # while describe_endpoint_response["EndpointStatus"] == "Updating":
        #     describe_endpoint_response = self.sm_client.describe_endpoint(
        #         EndpointName=endpoint_name
        #     )
        #     print(
        #         f"{describe_endpoint_response['EndpointStatus']} Endpoint: {endpoint_name}"
        #     )
        #     time.sleep(10)

        return describe_endpoint_response

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object"""
        return getattr(self.model, attr)


def handler(event, context=None):
    """Lambda handler to deploy a model to serverless endpoint"""

    model_name = event["model_name"]
    image_uri = event["image_uri"]
    endpoint_config_name = event["endpoint_config_name"]
    endpoint_name = event["endpoint_name"]
    memory_size_in_mb = int(event["memory_size_in_mb"])
    max_concurrency = int(event["max_concurrency"])

    # ----- Deploy model ----- #
    model = Model(name=model_name, image_uri=image_uri, role=role)

    model_adapted = ModelAdapter(model, sm_client)

    if check_endpoint_exists(endpoint_name):
        model_adapted.update_serverless_endpoint(
            endpoint_name=endpoint_name,
            endpoint_config_base_name=endpoint_config_name,
            memory_size_in_mb=memory_size_in_mb,
            max_concurrency=max_concurrency,
        )

    else:  # Create a new endpoint
        sic = ServerlessInferenceConfig(
            memory_size_in_mb=memory_size_in_mb, max_concurrency=max_concurrency
        )
        model_adapted.deploy(
            endpoint_name=endpoint_name, serverless_inference_config=sic
        )

    return {
        "statusCode": 200,
        "body": json.dumps("Created Endpoint!"),
    }
