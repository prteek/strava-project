FROM public.ecr.aws/lambda/python:3.9


COPY ./requirements.txt ./

COPY ./lam.py ./

RUN python3 -m pip install -r requirements.txt
