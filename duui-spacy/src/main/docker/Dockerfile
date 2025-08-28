FROM python:3.11

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install pip setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

# TODO include in requirements.txt
RUN pip install "benepar"

# languages/models
# - efficiency
RUN python -m spacy download "en_core_web_sm"
RUN python -m spacy download "de_core_news_sm"
RUN #python -m spacy download "xx_ent_wiki_sm"
#RUN python -m spacy download "ca_core_news_sm"
#RUN python -m spacy download "zh_core_web_sm"
#RUN python -m spacy download "hr_core_news_sm"
#RUN python -m spacy download "da_core_news_sm"
#RUN python -m spacy download "nl_core_news_sm"
#RUN python -m spacy download "fi_core_news_sm"
#RUN python -m spacy download "fr_core_news_sm"
#RUN python -m spacy download "el_core_news_sm"
#RUN python -m spacy download "it_core_news_sm"
#RUN python -m spacy download "ja_core_news_sm"
#RUN python -m spacy download "ko_core_news_sm"
#RUN python -m spacy download "lt_core_news_sm"
#RUN python -m spacy download "mk_core_news_sm"
#RUN python -m spacy download "nb_core_news_sm"
#RUN python -m spacy download "pl_core_news_sm"
#RUN python -m spacy download "pt_core_news_sm"
#RUN python -m spacy download "ro_core_news_sm"
#RUN python -m spacy download "ru_core_news_sm"
#RUN python -m spacy download "sl_core_news_sm"
#RUN python -m spacy download "es_core_news_sm"
#RUN python -m spacy download "sv_core_news_sm"
#RUN python -m spacy download "uk_core_news_sm"
# - accuracy
RUN python -m spacy download "en_core_web_trf"
RUN python -m spacy download "de_dep_news_trf"
RUN #python -m spacy download "xx_ent_wiki_sm"
#RUN python -m spacy download "ca_core_news_trf"
#RUN python -m spacy download "zh_core_web_trf"
#RUN python -m spacy download "hr_core_news_lg"
#RUN python -m spacy download "da_core_news_trf"
#RUN python -m spacy download "nl_core_news_lg"
#RUN python -m spacy download "fi_core_news_lg"
#RUN python -m spacy download "fr_dep_news_trf"
#RUN python -m spacy download "el_core_news_lg"
#RUN python -m spacy download "it_core_news_lg"
#RUN python -m spacy download "ja_core_news_trf"
#RUN python -m spacy download "ko_core_news_lg"
#RUN python -m spacy download "lt_core_news_lg"
#RUN python -m spacy download "mk_core_news_lg"
#RUN python -m spacy download "nb_core_news_lg"
#RUN python -m spacy download "pl_core_news_lg"
#RUN python -m spacy download "pt_core_news_lg"
#RUN python -m spacy download "ro_core_news_lg"
#RUN python -m spacy download "ru_core_news_lg"
#RUN python -m spacy download "sl_core_news_trf"
#RUN python -m spacy download "es_dep_news_trf"
#RUN python -m spacy download "sv_core_news_lg"
#RUN python -m spacy download "uk_core_news_trf"

# benepar models
RUN python -c "import benepar; benepar.download('benepar_en3')"
#RUN python -c "import benepar; benepar.download('benepar_en3_large')"
RUN python -c "import benepar; benepar.download('benepar_de2')"

# meta data
ARG TEXTIMAGER_SPACY_ANNOTATOR_NAME="textimager-duui-spacy"
ENV TEXTIMAGER_SPACY_ANNOTATOR_NAME=$TEXTIMAGER_SPACY_ANNOTATOR_NAME
ARG TEXTIMAGER_SPACY_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_SPACY_ANNOTATOR_VERSION=$TEXTIMAGER_SPACY_ANNOTATOR_VERSION

# log level
ARG TEXTIMAGER_SPACY_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_SPACY_LOG_LEVEL=$TEXTIMAGER_SPACY_LOG_LEVEL

# config
ARG TEXTIMAGER_SPACY_MODEL_CACHE_SIZE=3
ENV TEXTIMAGER_SPACY_MODEL_CACHE_SIZE=$TEXTIMAGER_SPACY_MODEL_CACHE_SIZE

# variant
ARG TEXTIMAGER_SPACY_VARIANT=""
ENV TEXTIMAGER_SPACY_VARIANT=$TEXTIMAGER_SPACY_VARIANT

# service script
COPY ./src/main/python/TypeSystemSpacy.xml ./TypeSystemSpacy.xml
COPY ./src/main/python/textimager_duui_spacy.py ./textimager_duui_spacy.py
COPY ./src/main/python/textimager_duui_spacy.lua ./textimager_duui_spacy.lua

ENTRYPOINT ["uvicorn", "textimager_duui_spacy:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]
