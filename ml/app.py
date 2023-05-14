#!/usr/bin/env python3

import aws_cdk as cdk
from ml_stacks import (
    ModelDeployerStack,
    StravaTrainProdPipelineStack,
    FitnessTrainProdPipelineStack,
)

app = cdk.App()
env = {"region": "eu-west-1"}

mls = ModelDeployerStack(app, "ModelDeployerStack", env=env)
pos = StravaTrainProdPipelineStack(app, "StravaTrainProdPipelineStack", env=env)
ftps = FitnessTrainProdPipelineStack(app, "FitnessTrainProdPipelineStack", env=env)

app.synth()
# %%
