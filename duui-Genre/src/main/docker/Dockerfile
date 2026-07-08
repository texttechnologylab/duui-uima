FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# copy scripts
COPY ./src/main/python/TypeSystemTopic.xml ./TypeSystemTopic.xml
COPY ./src/main/python/duui_genre.py ./duui_genre.py
COPY ./src/main/python/duui_genre.lua ./duui_genre.lua
COPY ./src/main/python/GenreSpeech.py ./GenreSpeech.py

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='TurkuNLP/web-register-classification-multilingual'); pipeline('fill-mask', model='FacebookAI/xlm-roberta-large')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='TurkuNLP/web-register-classification-en'); pipeline('fill-mask', model='FacebookAI/xlm-roberta-large')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='TurkuNLP/finerweb-quality-classifier')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ssharoff/genres')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='classla/xlm-roberta-base-multilingual-text-genre-classifier')"

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




ENTRYPOINT ["uvicorn", "duui_genre:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
