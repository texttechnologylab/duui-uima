FROM python:3.8

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

RUN python -m spacy download "en_core_web_trf"
RUN python -m spacy download "de_dep_news_trf"

RUN python -m spacy download "en_core_web_lg"
RUN python -m spacy download "de_core_news_lg"

RUN python -m spacy download "en_core_web_md"
RUN python -m spacy download "de_core_news_md"

RUN python -m spacy download "en_core_web_sm"
RUN python -m spacy download "de_core_news_sm"

ARG DUUI_SENTENCIZER_SPACY_LOG_LEVEL="DEBUG"
ENV DUUI_SENTENCIZER_SPACY_LOG_LEVEL=$DUUI_SENTENCIZER_SPACY_LOG_LEVEL

ARG DUUI_SENTENCIZER_SPACY_ANNOTATOR_NAME="duui-sentencizer-spacy"
ENV DUUI_SENTENCIZER_SPACY_ANNOTATOR_NAME=$DUUI_SENTENCIZER_SPACY_ANNOTATOR_NAME
ARG DUUI_SENTENCIZER_SPACY_ANNOTATOR_VERSION="unset"
ENV DUUI_SENTENCIZER_SPACY_ANNOTATOR_VERSION=$DUUI_SENTENCIZER_SPACY_ANNOTATOR_VERSION

ARG DUUI_SENTENCIZER_SPACY_MODEL_CACHE_SIZE=3
ENV DUUI_SENTENCIZER_SPACY_MODEL_CACHE_SIZE=$DUUI_SENTENCIZER_SPACY_MODEL_CACHE_SIZE

ARG DUUI_SENTENCIZER_SPACY_VARIANT=3
ENV DUUI_SENTENCIZER_SPACY_VARIANT=$DUUI_SENTENCIZER_SPACY_VARIANT

ARG DUUI_SENTENCIZER_SPACY_CUDA=-
ENV DUUI_SENTENCIZER_SPACY_CUDA=$DUUI_SENTENCIZER_SPACY_CUDA

COPY ./src/main/resources/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/python/duui.py ./duui.py
COPY ./src/main/lua/communication.lua ./communication.lua

ENTRYPOINT ["uvicorn", "duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
