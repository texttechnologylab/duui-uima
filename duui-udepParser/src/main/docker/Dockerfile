FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies

RUN pip install symspellpy fastapi uvicorn[standard] dkpro-cassis
RUN pip install setuptools wheel
#RUN pip install spacy
#RUN pip install hanta
RUN pip install diaparser


# languages/models
#RUN python -m spacy download "de_core_news_lg"
RUN python -c "from diaparser.parsers import Parser; Parser.load('de_hdt.dbmdz-bert-base')"
#RUN python -c "from HanTa import HanoverTagger as ht; ht.HanoverTagger('morphmodel_ger.pgz')"
# meta data
ARG TEXTIMAGER_UDEPPARSER_ANNOTATOR_NAME="textimager_duui_udepparser"
ENV TEXTIMAGER_UDEPPARSER_ANNOTATOR_NAME=$TEXTIMAGER_UDEPPARSER_ANNOTATOR_NAME
ARG TEXTIMAGER_UDEPPARSER_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_UDEPPARSER_ANNOTATOR_VERSION=$TEXTIMAGER_UDEPPARSER_ANNOTATOR_VERSION

# log level
ARG TEXTIMAGER_UDEPPARSER_LOG_LEVEL="INFO"
ENV TEXTIMAGER_UDEPPARSER_LOG_LEVEL=$TEXTIMAGER_UDEPPARSER_LOG_LEVEL

# config
ARG TEXTIMAGER_UDEPPARSER_PARSER_MODEL_NAME="de_hdt.dbmdz-bert-base"
ENV TEXTIMAGER_UDEPPARSER_PARSER_MODEL_NAME=$TEXTIMAGER_UDEPPARSER_MODEL_NAME

# offline mode for huggingface
ARG TEXTIMAGER_DUUI_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_OFFLINE

# service script
COPY ./src/main/python/TypeSystemUDEP.xml ./TypeSystemUDEP.xml
COPY ./src/main/python/textimager_duui_udep.lua ./textimager_duui_udep.lua
COPY ./src/main/python/textimager_duui_udep.py ./textimager_duui_udep.py


ENTRYPOINT ["uvicorn", "textimager_duui_udep:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]
