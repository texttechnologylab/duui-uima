FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

# log level
ARG log_level="DEBUG"
ENV log_level=$log_level

# meta data
ARG annotator_name="duui-vader-sentiment"
ENV annotator_name=$annotator_name
ARG annotator_version="unset"
ENV annotator_version=$annotator_version

# copy scripts
COPY ./src/main/resources/TypeSystemSentiment.xml ./src/main/resources/TypeSystemSentiment.xml
COPY ./src/main/python/__init__.py ./src/main/python/__init__.py
COPY ./src/main/python/duui/ ./src/main/python/duui/
COPY ./src/main/python/duui_vader_sentiment.py ./src/main/python/duui_vader_sentiment.py
COPY ./src/main/lua/duui_vader_sentiment.lua ./src/main/lua/duui_vader_sentiment.lua

ENTRYPOINT ["uvicorn", "src.main.python.duui_vader_sentiment:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
