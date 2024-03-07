FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714


# meta data
ARG FACT_ANNOTATOR_NAME="duui-Factchecking:app"
ENV FACT_ANNOTATOR_NAME=$FACT_ANNOTATOR_NAME
ARG FACT_ANNOTATOR_VERSION="unset"
ENV FACT_ANNOTATOR_VERSION=$FACT_ANNOTATOR_VERSION

# log level
ARG FACT_LOG_LEVEL="DEBUG"
ENV FACT_LOG_LEVEL=$FACT_LOG_LEVEL

# config
ARG FACT_MODEL_CACHE_SIZE=3
ENV FACT_MODEL_CACHE_SIZE=$FACT_MODEL_CACHE_SIZE

# Model Info
ARG FACT_MODEL_NAME=""
ENV FACT_MODEL_NAME=$FACT_MODEL_NAME
ARG FACT_MODEL_VERSION=0.1
ENV FACT_MODEL_VERSION=$FACT_MODEL_VERSION

# service script
COPY ./src/main/python/TypeSystemFactChecking.xml ./TypeSystemFactChecking.xml
COPY ./src/main/python/scorer.py ./scorer.py
COPY ./src/main/python/evaluator.py ./evaluator.py
COPY ./src/main/python/factchecker.py ./factchecker.py
COPY ./src/main/python/duui_fact.lua ./duui_fact.lua
COPY ./src/main/python/duui_fact.py ./duui_fact.py
COPY ./requirements.txt ./requirements.txt

RUN pip install --no-deps -r requirements.txt
RUN python -m nltk.downloader punkt



ENTRYPOINT ["uvicorn", "duui-FactChecking:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

