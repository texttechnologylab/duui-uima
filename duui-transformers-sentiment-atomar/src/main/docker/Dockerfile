FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='citizenlab/twitter-xlm-roberta-base-sentiment-finetunned')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='lxyuan/distilbert-base-multilingual-cased-sentiments-student')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='philschmid/distilbert-base-multilingual-cased-sentiment')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/sentiment-roberta-large-english-3-classes')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='bardsai/finance-sentiment-de-base')"

# copy scripts
COPY ./src/main/python/TypeSystemSentiment.xml ./TypeSystemSentiment.xml
COPY ./src/main/python/duui_transformers_sentiment.py ./duui_transformers_sentiment.py
COPY ./src/main/python/duui_transformers_sentiment.lua ./duui_transformers_sentiment.lua
COPY ./src/main/python/SentimentSpeech.py ./SentimentSpeech.py


# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config
ARG MODEL_CACHE_SIZE=3
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE

# meta data
ARG ANNOTATOR_NAME="duui-transformers-topic"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# Model Info
ARG MODEL_VERSION=0.1
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG=""
ENV MODEL_LANG=$MODEL_LANG

# offline mode for huggingface
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE


ENTRYPOINT ["uvicorn", "duui_transformers_sentiment:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
