FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /model

# copy & install requirements.txt
COPY deployml_course/requirements.txt /model/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY deployml_course/ /model

EXPOSE 6789

# run model with API
ENTRYPOINT [ "python" ] 
CMD [ "runner.py"] 



