FROM public.ecr.aws/lambda/python:3.9

COPY ./fetch_data.py ./
COPY ./predict_suffer_score.py ./
COPY ./predict_fitness_score.py ./

COPY ./requirements.txt ./
RUN python3 -m pip install -Ur requirements.txt
