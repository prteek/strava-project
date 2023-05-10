FROM python:3.10

COPY ./requirements.txt /opt/ml/processing/input/

RUN python3 -m pip install -U pip
RUN python3 -m pip install -Ur /opt/ml/processing/input/requirements.txt


COPY serve.py /usr/bin/serve
# required for serving
COPY helpers.py /usr/bin/helpers.py
COPY logger.py /usr/bin/logger.py

# Make serving script executable
RUN chmod 755 /usr/bin/serve

# Port for serving
EXPOSE 8080
