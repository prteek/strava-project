import os
from aws_cdk import aws_lambda, Duration, Stack
from constructs import Construct
from aws_cdk import aws_events, aws_events_targets


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
        self.func = aws_lambda.Function(
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

        lambda_schedule = aws_events.Schedule.cron(hour="4", minute="0", year="*", month="*", day="*")
        event_lambda_target = aws_events_targets.LambdaFunction(handler=self.func)
        lambda_cw_event = aws_events.Rule(
            self,
            id="fetch_strava_data_rule",
            description="The once per day CloudWatch event trigger for the Lambda",
            enabled=True,
            schedule=lambda_schedule,
            targets=[event_lambda_target])

