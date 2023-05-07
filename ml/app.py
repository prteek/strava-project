#!/usr/bin/env python3

import aws_cdk as cdk
from ml_stack import ModelDeployerStack, StravaTrainProdPipelineStack

app = cdk.App()
env = {"region": "eu-west-1"}

mls = ModelDeployerStack(app, "ModelDeployerStack", env=env)
pos = StravaTrainProdPipelineStack(app, "StravaTrainProdPipelineStack", env=env)

app.synth()
# %%
