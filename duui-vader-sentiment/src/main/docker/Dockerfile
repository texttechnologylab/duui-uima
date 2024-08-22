FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

# log level
ARG TEXTIMAGER_DUUI_VADER_SENTIMENT_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_DUUI_VADER_SENTIMENT_LOG_LEVEL=$TEXTIMAGER_DUUI_VADER_SENTIMENT_LOG_LEVEL

# meta data
ARG TEXTIMAGER_DUUI_VADER_SENTIMENT_ANNOTATOR_NAME="textimager-duui-vader-sentiment"
ENV TEXTIMAGER_DUUI_VADER_SENTIMENT_ANNOTATOR_NAME=$TEXTIMAGER_DUUI_VADER_SENTIMENT_ANNOTATOR_NAME
ARG TEXTIMAGER_DUUI_VADER_SENTIMENT_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_DUUI_VADER_SENTIMENT_ANNOTATOR_VERSION=$TEXTIMAGER_DUUI_VADER_SENTIMENT_ANNOTATOR_VERSION

# copy scripts
COPY ./src/main/python/gervader ./gervader
COPY ./src/main/resources/TypeSystemSentiment.xml ./src/main/resources/TypeSystemSentiment.xml
COPY ./src/main/python/__init__.py ./src/main/python/__init__.py
COPY ./src/main/python/duui/ ./src/main/python/duui/
COPY ./src/main/python/textimager_duui_vader_sentiment.py ./src/main/python/textimager_duui_vader_sentiment.py
COPY ./src/main/lua/textimager_duui_vader_sentiment.lua ./src/main/lua/textimager_duui_vader_sentiment.lua

ENTRYPOINT ["uvicorn", "src.main.python.textimager_duui_vader_sentiment:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
