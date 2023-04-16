import os
from aws_cdk import aws_lambda, Duration, Stack
from constructs import Construct
from aws_cdk import (aws_events,
                     aws_events_targets,
                     aws_apigateway as api_gw,
                     aws_iam as iam)

ENV = os.environ["ENV"]

architecture_map = {"dev": aws_lambda.Architecture.ARM_64,
                    "prod": aws_lambda.Architecture.X86_64}


class DataEngineeringStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create new Container Image.
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.path.join(os.getcwd(), "./engineering"),
            cmd=["fetch_data.handler"],
            file="engineering.Dockerfile",
        )

        # Lambda Function
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
        )

        event_policy = iam.PolicyStatement(effect=iam.Effect.ALLOW, resources=['*'], actions=['events:PutEvents'])
        data_fetch_lambda.add_to_role_policy(event_policy)


        lambda_schedule = aws_events.Schedule.cron(hour="4", minute="0", year="*", month="*", day="*")
        event_lambda_target = aws_events_targets.LambdaFunction(handler=data_fetch_lambda)
        lambda_cloudwatch_event = aws_events.Rule(
            self,
            id="fetch_strava_data_rule",
            description="The once per day CloudWatch event trigger for the Lambda",
            enabled=True,
            schedule=lambda_schedule,
            targets=[event_lambda_target])

        # Consumer Lambda
        # Create new Container Image.
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.path.join(os.getcwd(), "./engineering"),
            cmd=["predict_suffer_score.handler"],
            file="engineering.Dockerfile",
        )

        predictor_lambda = aws_lambda.Function(self,
                                                id="strava_predict_suffer_score",
                                                description="Make predictions on fetched data from Strava",
                                                code=ecr_image,
                                                handler=aws_lambda.Handler.FROM_IMAGE,
                                                runtime=aws_lambda.Runtime.FROM_IMAGE,
                                                architecture=architecture_map[ENV],
                                                function_name="strava_suffer_score_predictor",
                                                memory_size=128,
                                                reserved_concurrent_executions=2,
                                                timeout=Duration.seconds(120)
                                               )

        event_consumer_rule = aws_events.Rule(self,
                                              'eventConsumerLambdaRule',
                                              description='Predict on fetched activities from Strava',
                                              event_pattern=aws_events.EventPattern(source=['strava.activities']))

        event_consumer_rule.add_target(aws_events_targets.LambdaFunction(handler=predictor_lambda))
