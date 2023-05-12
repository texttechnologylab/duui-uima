FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies

RUN pip install symspellpy fastapi uvicorn[standard] dkpro-cassis
RUN pip install setuptools wheel
RUN pip install spacy
RUN pip install hanta
RUN pip install diaparser


# languages/models
RUN python -m spacy download "de_core_news_lg"
RUN python -c "from diaparser.parsers import Parser; Parser.load('de_hdt.dbmdz-bert-base')"
RUN python -c "from HanTa import HanoverTagger as ht; ht.HanoverTagger('morphmodel_ger.pgz')"
# meta data
ARG TEXTIMAGER_BFSRL_ANNOTATOR_NAME="textimager_duui_bfsrl"
ENV TEXTIMAGER_BFSRL_ANNOTATOR_NAME=$TEXTIMAGER_BFSRL_ANNOTATOR_NAME
ARG TEXTIMAGER_BFSRL_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_BFSRL_ANNOTATOR_VERSION=$TEXTIMAGER_BFSRL_ANNOTATOR_VERSION

# log level
ARG TEXTIMAGER_BFSRL_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_BFSRL_LOG_LEVEL=$TEXTIMAGER_BFSRL_LOG_LEVEL

# config
ARG TEXTIMAGER_BFSRL_PARSER_MODEL_NAME="de_hdt.dbmdz-bert-base"
ENV TEXTIMAGER_BFSRL_PARSER_MODEL_NAME=$TEXTIMAGER_BFSRL_MODEL_NAME

ARG TEXTIMAGER_SPACY_MODEL_CACHE_SIZE=3
ENV TEXTIMAGER_SPACY_MODEL_CACHE_SIZE=$TEXTIMAGER_SPACY_MODEL_CACHE_SIZE

# service script
COPY ./src/main/python/TypeSystemBFSRL.xml ./TypeSystemBFSRL.xml
#COPY ./src/main/python/uima_docker_wrapper_spacy.py ./uima_docker_wrapper_spacy.py
COPY ./src/main/python/xcomp-object.txt ./xcomp-object.txt
COPY ./src/main/python/xcomp-subject.txt ./xcomp-subject.txt
COPY ./src/main/python/textimager_duui_bfsrl.lua ./textimager_duui_bfsrl.lua
COPY ./src/main/python/textimager_duui_bfsrl.py ./textimager_duui_bfsrl.py
COPY ./src/main/python/bfsrl.py ./bfsrl.py


#ENTRYPOINT ["uvicorn", "uima_docker_wrapper_spacy:app", "--host", "0.0.0.0", "--port" ,"9714"]
ENTRYPOINT ["uvicorn", "textimager_duui_bfsrl:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]
