FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies

RUN pip install symspellpy fastapi uvicorn[standard] dkpro-cassis
RUN pip install setuptools wheel
RUN pip install numpy
RUN pip install hanta


# languages/models
RUN python -c "from HanTa import HanoverTagger as ht; ht.HanoverTagger('morphmodel_ger.pgz')"
# meta data
ARG TEXTIMAGER_HANTA_ANNOTATOR_NAME="textimager_duui_hanta"
ENV TEXTIMAGER_HANTA_ANNOTATOR_NAME=$TEXTIMAGER_HANTA_ANNOTATOR_NAME
ARG TEXTIMAGER_HANTA_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_HANTA_ANNOTATOR_VERSION=$TEXTIMAGER_HANTA_ANNOTATOR_VERSION

# log level
ARG TEXTIMAGER_HANTA_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_HANTA_LOG_LEVEL=$TEXTIMAGER_HANTA_LOG_LEVEL

# config
ARG TEXTIMAGER_HANTA_MODEL_NAME="morphmodel_ger.pgz"
ENV TEXTIMAGER_HANTA_MODEL_NAME=$TEXTIMAGER_HANTA_MODEL_NAME


# service script
COPY ./src/main/python/TypeSystemHANTA.xml ./TypeSystemHANTA.xml
#COPY ./src/main/python/uima_docker_wrapper_spacy.py ./uima_docker_wrapper_spacy.py
COPY ./src/main/python/textimager_duui_hanta.lua ./textimager_duui_hanta.lua
COPY ./src/main/python/textimager_duui_hanta.py ./textimager_duui_hanta.py


#ENTRYPOINT ["uvicorn", "uima_docker_wrapper_spacy:app", "--host", "0.0.0.0", "--port" ,"9714"]
ENTRYPOINT ["uvicorn", "textimager_duui_hanta:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]
