import os
from aws_cdk import aws_lambda, Duration, Stack, triggers, aws_iam
from constructs import Construct

ENV = os.environ["ENV"]

architecture_map = {"dev": aws_lambda.Architecture.ARM_64,
                    "prod": aws_lambda.Architecture.X86_64}

sagemaker_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
s3_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
lambda_exec_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambdaExecute")


class ModelDeployerStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create new Container Image.
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.path.join(os.getcwd(), "./ml"),
            cmd=["deploy_model.handler"],
            file="ml.Dockerfile",
        )

        role = aws_iam.Role(self, "strava_deployer_role",
                            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
                            managed_policies=[lambda_exec_policy, sagemaker_full_access_policy]
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
            environment={"SAGEMAKER_EXECUTION_ROLE": os.environ["SAGEMAKER_EXECUTION_ROLE"]},
            function_name="strava_model_deployer",
            memory_size=128,
            reserved_concurrent_executions=2,
            timeout=Duration.seconds(60),
            role=role
        )


class PipelineOrchestrationStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create new Container Image.
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.path.join(os.getcwd(), "./ml"),
            cmd=["orchestrate_pipeline.handler"],
            file="ml.Dockerfile",
        )

        role = aws_iam.Role(self, "strava_pipeline_orchestration_role",
                            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
                            managed_policies=[lambda_exec_policy,
                                              sagemaker_full_access_policy,
                                              s3_full_access_policy
                                              ]
                            )

        # Lambda Function
        orchestration_lambda = aws_lambda.Function(
            self,
            id="strava_pipeline_orchestrator",
            description="Orchestrate Strava ml pipeline on Sagemaker",
            code=ecr_image,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            architecture=architecture_map[ENV],
            environment={"SAGEMAKER_EXECUTION_ROLE": os.environ["SAGEMAKER_EXECUTION_ROLE"],
                         "IMAGE_URI": os.environ["IMAGE_URI"]},
            function_name="strava_pipeline_orchestrator",
            memory_size=128,
            reserved_concurrent_executions=2,
            timeout=Duration.seconds(120),
            role=role,
        )

        if ENV == "prod":
            # Trigger the function after it is deployed to orchestrate the pipeline
            trigger = triggers.Trigger(self, "start_orchestration_pipeline",
                                       handler=orchestration_lambda,
                                       execute_on_handler_change=True,
                                       )
        else:
            pass  # Do not update pipeline for dev automatically
