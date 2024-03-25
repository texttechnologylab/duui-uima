FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/python/TypeSystemComplexity.xml ./TypeSystemComplexity.xml
COPY ./src/main/python/BERT_converter.py ./BERT_converter.py
COPY ./src/main/python/Complexity.py ./Complexity.py
COPY ./src/main/python/duui_transformers_complexity.py ./duui_transformers_complexity.py
COPY ./src/main/python/duui_complexity.lua ./duui_complexity.lua

# dependencies
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -c "from transformers import pipeline; pipeline('fill-mask', model='intfloat/multilingual-e5-base')"
RUN python -c "from transformers import pipeline; pipeline('fill-mask', model='google-bert/bert-base-multilingual-cased')"
RUN python -c "from transformers import pipeline; pipeline('fill-mask', model='FacebookAI/xlm-roberta-large')"
RUN python -c "from transformers import pipeline; pipeline('fill-mask', model='facebook/xlm-v-base')"
RUN python -c "from transformers import pipeline; pipeline('fill-mask', model='cardiffnlp/twitter-xlm-roberta-base')"
RUN python -c "from transformers import pipeline; pipeline('fill-mask', model='setu4993/LEALLA-small')"
RUN python -c "from transformers import pipeline; pipeline('fill-mask', model='Twitter/twhin-bert-large')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('distiluse-base-multilingual-cased-v2')"

# log level
ARG COMPLEXITY_LOG_LEVEL="DEBUG"
ENV COMPLEXITY_LOG_LEVEL=$COMPLEXITY_LOG_LEVEL

# config
ARG COMPLEXITY_MODEL_CACHE_SIZE=3
ENV COMPLEXITY_MODEL_CACHE_SIZE=$COMPLEXITY_MODEL_CACHE_SIZE

# meta data
ARG COMPLEXITY_ANNOTATOR_NAME="duui-transformers-complexity"
ENV COMPLEXITY_ANNOTATOR_NAME=$COMPLEXITY_ANNOTATOR_NAME
ARG COMPLEXITY_ANNOTATOR_VERSION="unset"
ENV COMPLEXITY_ANNOTATOR_VERSION=$COMPLEXITY_ANNOTATOR_VERSION

# Model Info
#ARG COMPLEXITY_MODEL_NAME=""
#ENV COMPLEXITY_MODEL_NAME=$COMPLEXITY_MODEL_NAME
ARG COMPLEXITY_MODEL_VERSION=0.1
ENV COMPLEXITY_MODEL_VERSION=$COMPLEXITY_MODEL_VERSION
#ARG COMPLEXITY_MODEL_ART=""
#ENV COMPLEXITY_MODEL_ART=$COMPLEXITY_MODEL_ART
#ARG COMPLEXITY_MODEL_SOURCE=""
#ENV COMPLEXITY_MODEL_SOURCE=$COMPLEXITY_MODEL_SOURCE
#ARG COMPLEXITY_MODEL_LANG=""
#ENV COMPLEXITY_MODEL_LANG=$COMPLEXITY_MODEL_LANG

# offline mode for huggingface
#ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE




ENTRYPOINT ["uvicorn", "duui_transformers_complexity:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
