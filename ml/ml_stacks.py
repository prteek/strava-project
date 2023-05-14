import os
from aws_cdk import (
    aws_lambda,
    Duration,
    Stack,
    aws_iam,
    aws_sagemaker as sm,
    aws_events,
    aws_events_targets,
    aws_s3 as s3,
    aws_s3_notifications as s3_notify,
)
from constructs import Construct


ENV = os.environ["ENV"]
SM_ROLE = os.environ["SAGEMAKER_EXECUTION_ROLE"]
architecture_map = {
    "dev": aws_lambda.Architecture.ARM_64,
    "prod": aws_lambda.Architecture.X86_64,
}

sagemaker_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name(
    "AmazonSageMakerFullAccess"
)
s3_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name(
    "AmazonS3FullAccess"
)
lambda_exec_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name(
    "AWSLambdaExecute"
)


class ModelDeployerStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create new Container Image.
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.getcwd(),
            cmd=["deploy_model.handler"],
            file="ml.Dockerfile",
        )

        role = aws_iam.Role(
            self,
            "strava_deployer_role",
            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_exec_policy, sagemaker_full_access_policy],
        )

        # Lambda Function
        deployer_lambda = aws_lambda.Function(
            self,
            id="strava_model_deployer",
            description="Deploy Strava model to serverless endpoint",
            code=ecr_image,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            architecture=architecture_map[ENV],
            environment={"SAGEMAKER_EXECUTION_ROLE": SM_ROLE},
            function_name="strava_model_deployer",
            memory_size=128,
            reserved_concurrent_executions=2,
            timeout=Duration.seconds(60),
            role=role,
        )


class StravaTrainProdPipelineStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        with open("./training_pipeline_definition.json", encoding="utf8") as fp:
            pipeline_definition = fp.read()

        cfn_pipeline = sm.CfnPipeline(
            self,
            id="strava-ml-pipeline",
            pipeline_definition={
                "PipelineDefinitionBody": pipeline_definition,
                "PipelineDefinitionS3Location": None,
            },
            pipeline_name="strava-ml-pipeline",
            role_arn=SM_ROLE,
            # the properties below are optional
            pipeline_description="ML training pipeline",
            pipeline_display_name="strava-ml-pipeline",
        )

        # Invocation schedule lambda

        with open("./invoke_pipeline_lambda.py", encoding="utf8") as fp:
            handler_code = fp.read()

        role = aws_iam.Role(
            self,
            "strava_training_pipeline_invoke_role",
            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_exec_policy, sagemaker_full_access_policy],
        )

        lambdaFn = aws_lambda.Function(
            self,
            id="invoke_strava_training_pipeline",
            function_name="invoke_strava_training_pipeline",
            code=aws_lambda.InlineCode(handler_code),
            handler="index.invoke_training_pipeline_handler",
            timeout=Duration.seconds(60),
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            role=role,
        )

        rule = aws_events.Rule(
            self,
            "trigger_strava_training_pipeline_rule",
            schedule=aws_events.Schedule.rate(Duration.days(30)),
            targets=[aws_events_targets.LambdaFunction(lambdaFn)],
            enabled=True,
        )


class FitnessTrainProdPipelineStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        with open("./fitness_training_pipeline_definition.json", encoding="utf8") as fp:
            pipeline_definition = fp.read()

        cfn_pipeline = sm.CfnPipeline(
            self,
            id="fitness-ml-pipeline",
            pipeline_definition={
                "PipelineDefinitionBody": pipeline_definition,
                "PipelineDefinitionS3Location": None,
            },
            pipeline_name="fitness-ml-pipeline",
            role_arn=SM_ROLE,
            # the properties below are optional
            pipeline_description="Fitness ML training pipeline",
            pipeline_display_name="fitness-ml-pipeline",
        )

        # Invocation schedule lambda

        with open("./invoke_pipeline_lambda.py", encoding="utf8") as fp:
            handler_code = fp.read()

        role = aws_iam.Role(
            self,
            "fitness_training_pipeline_invoke_role",
            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_exec_policy, sagemaker_full_access_policy],
        )

        lambdaFn = aws_lambda.Function(
            self,
            id="invoke_fitness_training_pipeline",
            function_name="invoke_fitness_training_pipeline",
            code=aws_lambda.InlineCode(handler_code),
            handler="index.invoke_fitness_training_pipeline_handler",
            timeout=Duration.seconds(60),
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            role=role,
        )

        # Create S3 Bucket
        bucket = s3.Bucket.from_bucket_name(
            self, id="strava-bucket", bucket_name="pp-strava-data"
        )

        # Create trigger for Lambda function using suffix
        notification = s3_notify.LambdaDestination(lambdaFn)
        notification.bind(self, bucket)

        # Add Create Event only for fitness csv
        bucket.add_object_created_notification(
            notification,
            s3.NotificationKeyFilter(prefix="fitness-curve-data", suffix=".csv"),
        )
