FROM python:3.13

WORKDIR /usr/src/app

EXPOSE 9714y

RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

ARG DUUI_SENTENCE_TRANSFORMERS_LOG_LEVEL="DEBUG"
ENV DUUI_SENTENCE_TRANSFORMERS_LOG_LEVEL=$DUUI_SENTENCE_TRANSFORMERS_LOG_LEVEL

ARG DUUI_SENTENCE_TRANSFORMERS_MODEL_CACHE_SIZE=1
ENV DUUI_SENTENCE_TRANSFORMERS_MODEL_CACHE_SIZE=$DUUI_SENTENCE_TRANSFORMERS_MODEL_CACHE_SIZE

ARG DUUI_SENTENCE_TRANSFORMERS_ANNOTATOR_NAME="duui-sentence-transformers"
ENV DUUI_SENTENCE_TRANSFORMERS_ANNOTATOR_NAME=$DUUI_SENTENCE_TRANSFORMERS_ANNOTATOR_NAME
ARG DUUI_SENTENCE_TRANSFORMERS_ANNOTATOR_VERSION="unset"
ENV DUUI_SENTENCE_TRANSFORMERS_ANNOTATOR_VERSION=$DUUI_SENTENCE_TRANSFORMERS_ANNOTATOR_VERSION

COPY ./src/main/resources/TypeSystem.xml ./src/main/resources/TypeSystem.xml
COPY ./src/main/python/duui.py ./src/main/python/duui.py
COPY ./src/main/lua/communication.lua ./src/main/lua/communication.lua

ENTRYPOINT ["uvicorn", "src.main.python.duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
