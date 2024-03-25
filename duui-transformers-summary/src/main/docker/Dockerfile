FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -c "from transformers import pipeline; pipeline('text2text-generation', model='csebuetnlp/mT5_multilingual_XLSum')"
RUN python -c "from transformers import pipeline; pipeline('text2text-generation', model='google/flan-t5-base')"
RUN python -c "from mdmls import Summarizer; Summarizer(device=-1)"

# copy scripts
COPY ./src/main/python/TypeSystemSummary.xml ./TypeSystemSummary.xml
COPY ./src/main/python/duui_summary.lua ./duui_summary.lua
COPY ./src/main/python/summarization.py ./summarization.py
COPY ./src/main/python/duui_transformers_summary.py ./duui_transformers_summary.py


# log level
ARG SUMMARY_LOG_LEVEL="DEBUG"
ENV SUMMARY_LOG_LEVEL=$SUMMARY_LOG_LEVEL

# config
ARG SUMMARY_MODEL_CACHE_SIZE=3
ENV SUMMARY_MODEL_CACHE_SIZE=$SUMMARY_MODEL_CACHE_SIZE

# meta data
ARG SUMMARY_ANNOTATOR_NAME="duui-transformers-complexity"
ENV SUMMARY_ANNOTATOR_NAME=$SUMMARY_ANNOTATOR_NAME
ARG SUMMARY_ANNOTATOR_VERSION="unset"
ENV SUMMARY_ANNOTATOR_VERSION=$SUMMARY_ANNOTATOR_VERSION

# Model Info
ARG SUMMARY_MODEL_VERSION=0.1
ENV SUMMARY_MODEL_VERSION=$SUMMARY_MODEL_VERSION


ENTRYPOINT ["uvicorn", "duui_transformers_summary:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
