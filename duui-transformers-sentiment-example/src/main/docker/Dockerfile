FROM python:3.8

WORKDIR /usr/src/app

# This port is specified by the DUUI API
EXPOSE 9714

# Install dependencies
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

# Preload Models during Docker build
RUN python3 -c 'from transformers import pipeline; pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment", revision="f3e34b6c30bf27b6649f72eca85d0bbe79df1e55")'

# Log level
ARG TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_LOG_LEVEL="DEBUG"
ENV TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_LOG_LEVEL=$TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_LOG_LEVEL

# Meta data
ARG TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_ANNOTATOR_NAME="duui-transformers-sentiment-example"
ENV TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_ANNOTATOR_NAME=$TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_ANNOTATOR_NAME
ARG TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_ANNOTATOR_VERSION="dev"
ENV TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_ANNOTATOR_VERSION=$TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_ANNOTATOR_VERSION

# Enable offline mode for Hugging Face
ARG TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_TRANSFORMERS_OFFLINE

# Copy all needed scripts
COPY ./src/main/resources/TypeSystemSentiment.xml ./src/main/resources/TypeSystemSentiment.xml
COPY ./src/main/python/duui_transformers_sentiment.py ./src/main/python/duui_transformers_sentiment.py
COPY ./src/main/lua/duui_transformers_sentiment.lua ./src/main/lua/duui_transformers_sentiment.lua

# Start the Python server
ENTRYPOINT ["uvicorn", "src.main.python.duui_transformers_sentiment:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
