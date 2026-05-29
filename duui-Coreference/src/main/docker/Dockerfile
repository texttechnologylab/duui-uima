FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python3 -m coreferee install en
RUN python3 -m coreferee install de
RUN python3 -m coreferee install fr
RUN python3 -m coreferee install pl
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm
RUN python -m spacy download fr_core_news_sm
RUN python -m spacy download pl_core_news_sm

# copy scripts
COPY ./src/main/python/TypeSystemCoreference.xml ./TypeSystemCoreference.xml
COPY ./src/main/python/duui_coreference.py ./duui_coreference.py
COPY ./src/main/python/duui_coreference.lua ./duui_coreference.lua
#COPY ./src/main/python/GenreSpeech.py ./GenreSpeech.py

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config
ARG MODEL_CACHE_SIZE=3
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE

# meta data
ARG ANNOTATOR_NAME="duui-transformers-topic"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# Model Info
ARG MODEL_VERSION=0.1
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG=""
ENV MODEL_LANG=$MODEL_LANG
ARG MODEL_VARIANT=""
ENV MODEL_VARIANT=$MODEL_VARIANT


# offline mode for huggingface
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE




ENTRYPOINT ["uvicorn", "duui_coreference:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
