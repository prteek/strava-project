import os
from aws_cdk import aws_lambda, Duration, Stack
from constructs import Construct


class DataEngineeringStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create new Container Image.
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.path.join(os.getcwd(), "./engineering"), cmd=["fetch_data.handler"]
        )

        # Lambda Function
        self.lam = aws_lambda.Function(
            self,
            id="fetch_data_strava",
            description="Fetch data from Strava",
            code=ecr_image,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            architecture=aws_lambda.Architecture.ARM_64,
            environment={"STRAVA_CLIENT_ID": os.environ["STRAVA_CLIENT_ID"],
                         "STRAVA_CLIENT_SECRET": os.environ["STRAVA_CLIENT_SECRET"],
                         "STRAVA_ACCESS_TOKEN": os.environ["STRAVA_ACCESS_TOKEN"],
                         "STRAVA_REFRESH_TOKEN": os.environ["STRAVA_REFRESH_TOKEN"],
                         "EXPIRES_AT": os.environ["EXPIRES_AT"]
                         },
            function_name="fetchStravaData",
            memory_size=256,
            reserved_concurrent_executions=2,
            timeout=Duration.seconds(120),
        )
