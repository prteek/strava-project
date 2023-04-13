FROM public.ecr.aws/lambda/python:3.9

COPY ./model_deployer.py ./
COPY ./config.txt ./
COPY ./orchestrate.py ./

RUN python3 -m pip install -U sagemaker 'sagemaker[local]'