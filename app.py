#!/usr/bin/env python3

import aws_cdk as cdk
from lambda_functions.lambda_stack import LambdaFunctionStack
from ml.ml_lambda_stack import ModelDeployerStack, PipelineOrchestrationStack

app = cdk.App()
env = {"region": "eu-west-1"}

lfs = LambdaFunctionStack(app, "LambdaFunctionStack", env=env)
mls = ModelDeployerStack(app, "ModelDeployerStack", env=env)
pos = PipelineOrchestrationStack(app, "PipelineOrchestrationStack", env=env)

app.synth()
# %%
