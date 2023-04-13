FROM public.ecr.aws/lambda/python:3.9

COPY ./model_deployer.py ./

RUN python3 -m pip install -U sagemaker 'sagemaker[local]'