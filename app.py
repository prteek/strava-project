#!/usr/bin/env python3

import aws_cdk as cdk
from lambda_functions.lambda_stack import LambdaFunctionStack


app = cdk.App()
env = {"region": "eu-west-1"}

lcf = LambdaFunctionStack(app, "LambdaFunctionStack", env=env)

app.synth()
# %%
