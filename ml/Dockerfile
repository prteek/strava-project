FROM python:3.10

COPY ./requirements.txt /opt/ml/processing/input/

RUN python3 -m pip install -U pip
RUN python3 -m pip install -Ur /opt/ml/processing/input/requirements.txt

COPY helpers.py /usr/bin/helpers.py
COPY logger.py /usr/bin/logger.py

