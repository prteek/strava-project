FROM public.ecr.aws/lambda/python:3.9

# Common dependencies
RUN python3 -m pip install -U sagemaker

# Model deployer lambda
COPY ./deploy_model.py ./

