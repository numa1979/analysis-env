FROM python:3.10-slim-buster

WORKDIR /opt/build
ADD requirements.txt /opt/build/
RUN pip install -U pip ipython jupyterlab_code_formatter

RUN pip install -r requirements.txt