FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt
RUN python -m spacy download en_core_web_sm

# service script
COPY ./src/main/python/TypeSystemTextReadability.xml ./TypeSystemTextReadability.xml
COPY ./src/main/python/Readability.py ./Readability.py
COPY ./src/main/python/duui_readability.lua ./duui_readability.lua
COPY ./src/main/python/duui_readability.py ./duui_readability.py


# meta data
ARG ANNOTATOR_NAME="duui-readability:app"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config
ARG MODEL_CACHE_SIZE=1
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_VERSION=""
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG=""
ENV MODEL_LANG=$MODEL_LANG

# offline mode for huggingface
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE



ENTRYPOINT ["uvicorn", "duui_readability:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

