import os
from aws_cdk import aws_lambda, Duration, Stack
from constructs import Construct


class ModelDeployerStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create new Container Image.
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.path.join(os.getcwd(), "./ml"),
            cmd=["model_deployer.handler"],
            file="ml.Dockerfile",
        )

        # Lambda Function
        self.lam = aws_lambda.Function(
            self,
            id="strava_model_deployer",
            description="Deploy Strava model to serverless endpoint",
            code=ecr_image,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            architecture=aws_lambda.Architecture.ARM_64,
            environment={"SAGEMAKER_EXECUTION_ROLE": os.environ["SAGEMAKER_EXECUTION_ROLE"]},
            function_name="deployStravaModel",
            memory_size=128,
            reserved_concurrent_executions=2,
            timeout=Duration.seconds(60),
        )


class PipelineOrchestrationStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create new Container Image.
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.path.join(os.getcwd(), "./ml"),
            cmd=["orchestrate.handler"],
            file="ml.Dockerfile",
        )

        # Lambda Function
        self.lam = aws_lambda.Function(
            self,
            id="strava_pipeline_orchestrator",
            description="Orchestrate Strava ml pipeline on Sagemaker",
            code=ecr_image,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            architecture=aws_lambda.Architecture.ARM_64,
            environment={"SAGEMAKER_EXECUTION_ROLE": os.environ["SAGEMAKER_EXECUTION_ROLE"],
                         "IMAGE_URI": os.environ["IMAGE_URI"]},
            function_name="orchestrateStravaPipeline",
            memory_size=128,
            reserved_concurrent_executions=2,
            timeout=Duration.seconds(60),
        )
