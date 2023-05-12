FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies

RUN pip install symspellpy fastapi uvicorn[standard] dkpro-cassis
RUN pip install setuptools wheel
RUN pip install pathos omegaconf
#RUN pip install hanta
#RUN pip install diaparser

RUN git clone https://github.com/yzhangcs/crfsrl.git --recursive

# languages/models
#RUN python -m spacy download "de_core_news_lg"
#RUN python -c "from diaparser.parsers import Parser; Parser.load('de_hdt.dbmdz-bert-base')"
#RUN python -c "from HanTa import HanoverTagger as ht; ht.HanoverTagger('morphmodel_ger.pgz')"
# meta data
ARG TEXTIMAGER_SRL_ANNOTATOR_NAME="textimager_duui_srl"
ENV TEXTIMAGER_SRL_ANNOTATOR_NAME=$TEXTIMAGER_SRL_ANNOTATOR_NAME
ARG TEXTIMAGER_SRL_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_SRL_ANNOTATOR_VERSION=$TEXTIMAGER_SRL_ANNOTATOR_VERSION

# log level
ARG TEXTIMAGER_SRL_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_SRL_LOG_LEVEL=$TEXTIMAGER_SRL_LOG_LEVEL

# config
ARG TEXTIMAGER_SRL_PARSER_MODEL_NAME="de_hdt.dbmdz-bert-base"
ENV TEXTIMAGER_SRL_PARSER_MODEL_NAME=$TEXTIMAGER_SRL_MODEL_NAME

# offline mode for huggingface
ARG TEXTIMAGER_DUUI_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_OFFLINE
# service script
COPY ./src/main/python/TypeSystemSRL.xml ./TypeSystemSRL.xml
COPY ./src/main/python/TypeSystemSRL.xml ./TypeSystemSRL.xml
#COPY ./src/main/python/uima_docker_wrapper_spacy.py ./uima_docker_wrapper_spacy.py
COPY ./src/main/python/textimager_duui_srl.lua ./textimager_duui_srl.lua
COPY ./src/main/python/textimager_duui_srl.py ./textimager_duui_srl.py


#ENTRYPOINT ["uvicorn", "uima_docker_wrapper_spacy:app", "--host", "0.0.0.0", "--port" ,"9714"]
ENTRYPOINT ["uvicorn", "textimager_duui_srl:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]
