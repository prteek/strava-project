FROM public.ecr.aws/lambda/python:3.9

# Common dependencies
RUN python3 -m pip install -U sagemaker 'sagemaker[local]'

# Model deployer lambda
COPY ./deploy_model.py ./

# Pipeline orchestrator lambda
COPY ./orchestrate_pipeline.py ./
COPY ./config.txt ./
COPY ./logger.py ./
COPY ./helpers.py ./
COPY ./train_model.py ./
COPY ./prepare_training_data.py ./
