#!/usr/bin/env python3

import aws_cdk as cdk
from engineering.engineering_stack import DataEngineeringStack
from ml.ml_lambda_stack import ModelDeployerStack, PipelineOrchestrationStack

app = cdk.App()
env = {"region": "eu-west-1"}

lfs = DataEngineeringStack(app, "DataEngineeringStack", env=env)
mls = ModelDeployerStack(app, "ModelDeployerStack", env=env)
pos = PipelineOrchestrationStack(app, "PipelineOrchestrationStack", env=env)

app.synth()
# %%
