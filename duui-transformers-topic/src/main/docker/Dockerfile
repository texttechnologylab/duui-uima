FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/tweet-topic-latest-multi')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='classla/xlm-roberta-base-multilingual-text-genre-classifier')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='chkla/parlbert-topic-german')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ssharoff/genres')"


# log level
ARG TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_LOG_LEVEL=$TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_LOG_LEVEL

# config
ARG TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_MODEL_CACHE_SIZE=1
ENV TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_MODEL_CACHE_SIZE=$TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_MODEL_CACHE_SIZE

# meta data
ARG TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_ANNOTATOR_NAME="textimager-duui-transformers-topic"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_ANNOTATOR_NAME=$TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_ANNOTATOR_NAME
ARG TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_ANNOTATOR_VERSION=$TEXTIMAGER_DUUI_TRANSFORMERS_TOPIC_ANNOTATOR_VERSION

# offline mode for huggingface
#ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE

# copy scripts
COPY ./src/main/resources/TypeSystemTopic.xml ./src/main/resources/TypeSystemTopic.xml
COPY ./src/main/python/__init__.py ./src/main/python/__init__.py
COPY ./src/main/python/duui/ ./src/main/python/duui/
COPY ./src/main/python/textimager_duui_transformers_topic.py ./src/main/python/textimager_duui_transformers_topic.py
COPY ./src/main/lua/textimager_duui_transformers_topic.lua ./src/main/lua/textimager_duui_transformers_topic.lua

ENTRYPOINT ["uvicorn", "src.main.python.textimager_duui_transformers_topic:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
