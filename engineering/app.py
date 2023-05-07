#!/usr/bin/env python3

import aws_cdk as cdk
from engineering_stacks import DataEngineeringStack

app = cdk.App()
env = {"region": "eu-west-1"}

lfs = DataEngineeringStack(app, "DataEngineeringStack", env=env)

app.synth()
# %%
