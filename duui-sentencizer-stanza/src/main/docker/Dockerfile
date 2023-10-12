FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

RUN python -c "import stanza; stanza.download('en')"
RUN python -c "import stanza; stanza.download('de')"

ARG DUUI_SENTENCIZER_STANZA_LOG_LEVEL="DEBUG"
ENV DUUI_SENTENCIZER_STANZA_LOG_LEVEL=$DUUI_SENTENCIZER_STANZA_LOG_LEVEL

ARG DUUI_SENTENCIZER_STANZA_ANNOTATOR_NAME="duui-sentencizer-stanza"
ENV DUUI_SENTENCIZER_STANZA_ANNOTATOR_NAME=$DUUI_SENTENCIZER_STANZA_ANNOTATOR_NAME
ARG DUUI_SENTENCIZER_STANZA_ANNOTATOR_VERSION="unset"
ENV DUUI_SENTENCIZER_STANZA_ANNOTATOR_VERSION=$DUUI_SENTENCIZER_STANZA_ANNOTATOR_VERSION

ARG DUUI_SENTENCIZER_STANZA_MODEL_CACHE_SIZE=3
ENV DUUI_SENTENCIZER_STANZA_MODEL_CACHE_SIZE=$DUUI_SENTENCIZER_STANZA_MODEL_CACHE_SIZE

COPY ./src/main/resources/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/python/duui.py ./duui.py
COPY ./src/main/lua/communication.lua ./communication.lua

ENTRYPOINT ["uvicorn", "duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
