import os
import sys
import json
import argparse
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn import SKLearn
from sagemaker.model import Model
import configparser
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.lambda_step import (
    LambdaStep,
)
from sagemaker.lambda_helper import Lambda
from datetime import datetime

config = configparser.ConfigParser()
config.read("config.txt")
role = os.environ["SAGEMAKER_EXECUTION_ROLE"]


def upload_code_helpers(filepath_list: list, bucket: str, prefix: str) -> str:
    for filepath in filepath_list:
        _ = sagemaker.Session().upload_data(filepath, bucket, key_prefix=prefix)

    return f"s3://{bucket}/{prefix}/"


def create_pipeline(on_aws=False):
    """Create a pipeline for training and deploying a model"""

    if on_aws:
        processor_instance_type = "ml.t3.medium"
        train_instance_type = "ml.m5.large"
        session = PipelineSession()
    else:
        processor_instance_type = "local"
        train_instance_type = "local"
        session = LocalPipelineSession()

    bucket = config.get("aws", "bucket")
    image_uri = os.environ["IMAGE_URI"]
    processing_local_dependencies = ["logger.py", "helpers.py"]
    helpers = upload_code_helpers(
        processing_local_dependencies, bucket, prefix="helpers"
    )

    # ----- Prepare training data ----- #
    prepare_data_code_location = session.upload_data(
        "prepare_training_data.py", bucket=bucket, key_prefix="prepare-training-data/code"
    )

    prepare_data_output = f"s3://{bucket}/prepare-training-data/output/"

    prepare_data = ScriptProcessor(
        base_job_name="prepare-data",
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_count=1,
        instance_type=processor_instance_type,
        sagemaker_session=session,
    )

    prepare_data_step = ProcessingStep(
        name="prepare-training-data",
        processor=prepare_data,
        outputs=[
            ProcessingOutput(
                output_name="train",
                destination=prepare_data_output,
                source="/opt/ml/processing/output/",
            )],
        inputs=[ProcessingInput(helpers, "/opt/ml/processing/input")],
        code=prepare_data_code_location,
    )

    # ----- Train model ----- #
    train_data_location = prepare_data_output
    train_output_location = (
        f"s3://{bucket}/train/job-artefacts"  # Model artefacts will be uploaded here
    )
    local_dependencies = ["logger.py", "helpers.py"]

    estimator = SKLearn(
        base_job_name="model-training",
        role=role,
        entry_point="train_model.py",
        framework_version="1.0-1",
        instance_count=1,
        instance_type=train_instance_type,
        dependencies=local_dependencies,
        code_location=train_output_location,
        output_path=train_output_location,
        sagemaker_session=session,
    )

    train_step = TrainingStep(name='train-model',
                              estimator=estimator,
                              inputs={"train": TrainingInput(s3_data=train_data_location)},
                              depends_on=[prepare_data_step])

    # ----- Create model ----- #
    model_name = f"{config.get('model', 'name')}-{datetime.now().strftime('%Y%m%d')}"
    code_location = f"s3://{bucket}/train/Model"  # Code files will be uploaded here

    model = Model(
        image_uri=estimator.image_uri,
        sagemaker_session=session,
        role=role,
        name=model_name,
        entry_point="train_model.py",
        code_location=code_location,
        dependencies=local_dependencies,
    )

    model_step_args = model.create(instance_type=train_instance_type)

    model_step = ModelStep(
        name="model-step",
        step_args=model_step_args,
        depends_on=[train_step],
    )

    if not on_aws:
        # Define the pipeline in local mode and execute
        pipeline = Pipeline(name='local-pipeline',
                            steps=[prepare_data_step, train_step, model_step],
                            sagemaker_session=session)

        return pipeline

    else:

        # ----- Deploy model ----- #
        memory_size_in_mb = config.get("endpoint", "memory-size-in-mb")
        max_concurrency = config.get("endpoint", "max-concurrency")
        endpoint_config_name = config.get("endpoint", "config-name")
        endpoint_name = config.get("endpoint", "name")
        deployer_lambda_arn = config.get("aws", "deployer-lambda-arn")

        deployer_lambda = Lambda(deployer_lambda_arn)
        deploy_step = LambdaStep(
            name="deploy-model",
            lambda_func=deployer_lambda,
            inputs={
                "model_name": model_name,
                "endpoint_config_name": endpoint_config_name,
                "endpoint_name": endpoint_name,
                "image_uri": estimator.image_uri,
                "role": role,
                "memory_size_in_mb": memory_size_in_mb,
                "max_concurrency": max_concurrency,
            },
            depends_on=[model_step],
        )

        # Define the pipeline on aws but do not execute (since this process will be executed on a lambda)
        pipeline = Pipeline(name='strava-ml-pipeline',
                            steps=[prepare_data_step, train_step, model_step, deploy_step],
                            sagemaker_session=session)

        return pipeline


def handler(event, context=None):
    """Lambda handler to create the pipeline on aws"""

    pipeline = create_pipeline(on_aws=True)
    pipeline.upsert(role_arn=role, description='Setup deployment pipeline')

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Pipeline orchestrated successfully"})
    }


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

    pipeline = create_pipeline(on_aws=on_aws)
    if not on_aws: # Halt script execution here if this is a test
        pipeline.upsert(role_arn=role, description='local pipeline execution')
        # Start a pipeline execution
        execution = pipeline.start()

        print("Local orchestration test complete")
        sys.exit(0)
    else:
        pipeline.upsert(role_arn=role, description='Setup deployment pipeline')
