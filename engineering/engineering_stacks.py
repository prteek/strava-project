import os
from constructs import Construct
from aws_cdk import (aws_lambda,
                     Duration,
                     Stack,
                     aws_events,
                     aws_events_targets,
                     aws_lambda_destinations,
                     aws_iam
                     )

ENV = os.environ["ENV"]

architecture_map = {"dev": aws_lambda.Architecture.ARM_64,
                    "prod": aws_lambda.Architecture.X86_64}


lambda_exec_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambdaExecute")
sagemaker_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
s3_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
athena_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonAthenaFullAccess")
event_bridge_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEventBridgeFullAccess")
glue_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AWSGlueConsoleFullAccess")


class DataEngineeringStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Fitness Lambda
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.getcwd(),
            cmd=["predict_fitness_score.handler"],
            file="engineering.Dockerfile",
        )

        role = aws_iam.Role(self, "strava_fitness_predictor_lambda_role",
                            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
                            managed_policies=[lambda_exec_policy,
                                              s3_full_access_policy,
                                              athena_full_access_policy,
                                              sagemaker_full_access_policy,
                                              glue_full_access_policy,
                                              ]
                            )

        fitness_score_lambda = aws_lambda.Function(self,
                                               id="strava_predict_fitness_score",
                                               description="Make predictions using suffer score data from Strava",
                                               code=ecr_image,
                                               handler=aws_lambda.Handler.FROM_IMAGE,
                                               runtime=aws_lambda.Runtime.FROM_IMAGE,
                                               architecture=architecture_map[ENV],
                                               function_name="strava_fitness_score_predictor",
                                               memory_size=256,
                                               reserved_concurrent_executions=2,
                                               timeout=Duration.seconds(600),
                                               role=role,
                                               )

        # Suffer score Lambda
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.getcwd(),
            cmd=["predict_suffer_score.handler"],
            file="engineering.Dockerfile",
        )

        role = aws_iam.Role(self, "strava_predictor_lambda_role",
                            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
                            managed_policies=[lambda_exec_policy,
                                              s3_full_access_policy,
                                              athena_full_access_policy,
                                              sagemaker_full_access_policy,
                                              glue_full_access_policy,
                                              ]
                            )

        suffer_score_lambda = aws_lambda.Function(self,
                                               id="strava_predict_suffer_score",
                                               description="Make predictions on fetched data from Strava",
                                               code=ecr_image,
                                               handler=aws_lambda.Handler.FROM_IMAGE,
                                               runtime=aws_lambda.Runtime.FROM_IMAGE,
                                               architecture=architecture_map[ENV],
                                               function_name="strava_suffer_score_predictor",
                                               memory_size=256,
                                               reserved_concurrent_executions=2,
                                               timeout=Duration.seconds(180),
                                               on_success=aws_lambda_destinations.LambdaDestination(fitness_score_lambda,
                                                                                                    response_only=True),
                                               role=role,
                                               )


        # Data Fetch Lambda
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.getcwd(),
            cmd=["fetch_data.handler"],
            file="engineering.Dockerfile",
        )

        role = aws_iam.Role(self, "strava_fetch_data_lambda_role",
                            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
                            managed_policies=[lambda_exec_policy,
                                              s3_full_access_policy,
                                              athena_full_access_policy,
                                              event_bridge_full_access_policy,
                                              glue_full_access_policy,
                                             ]
                            )

        data_fetch_lambda = aws_lambda.Function(
            self,
            id="fetch_data_strava",
            description="Fetch data from Strava",
            code=ecr_image,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            architecture=architecture_map[ENV],
            environment={
                "STRAVA_CLIENT_ID": os.environ["STRAVA_CLIENT_ID"],
                "STRAVA_CLIENT_SECRET": os.environ["STRAVA_CLIENT_SECRET"],
                "STRAVA_ACCESS_TOKEN": os.environ["STRAVA_ACCESS_TOKEN"],
                "STRAVA_REFRESH_TOKEN": os.environ["STRAVA_REFRESH_TOKEN"],
                "EXPIRES_AT": os.environ["EXPIRES_AT"],
            },
            function_name="strava_data_fetcher",
            memory_size=256,
            reserved_concurrent_executions=2,
            timeout=Duration.seconds(120),
            on_success=aws_lambda_destinations.LambdaDestination(suffer_score_lambda,
                                                                 response_only=True),
            role=role
        )

        lambda_schedule = aws_events.Schedule.cron(hour="4", minute="0", year="*", month="*", day="*")
        event_lambda_target = aws_events_targets.LambdaFunction(handler=data_fetch_lambda)
        lambda_cloudwatch_event = aws_events.Rule(
            self,
            id="fetch_strava_data_rule",
            description="The once per day CloudWatch event trigger for the Lambda",
            enabled=True,
            schedule=lambda_schedule,
            targets=[event_lambda_target])

