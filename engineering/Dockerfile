FROM public.ecr.aws/lambda/python:3.9

COPY ./lam.py ./

COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt
