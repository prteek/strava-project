FROM public.ecr.aws/lambda/python:3.9

# Common dependencies
RUN python3 -m pip install -U sagemaker 'sagemaker[local]'

# Model deployer lambda
COPY ./model_deployer.py ./

# Pipeline orchestrator lambda
COPY ./orchestrate.py ./
COPY ./config.txt ./
COPY ./logger.py ./
COPY ./helpers.py ./
COPY ./train.py ./
COPY ./prepare_training_data.py ./
