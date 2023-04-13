FROM public.ecr.aws/lambda/python:3.9

RUN python3 -m pip install -U sagemaker 'sagemaker[local]'