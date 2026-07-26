FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/python/TypeSystemTopic.xml ./TypeSystemTopic.xml
COPY ./src/main/python/duui_transformers_topic.py ./duui_transformers_topic.py
COPY ./src/main/python/duui_transformers_topic.lua ./duui_transformers_topic.lua
COPY ./src/main/python/TopicSpeech.py ./TopicSpeech.py

# dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='KnutJaegersberg/topic-classification-IPTC-subject-labels')"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('poltextlab/xlm-roberta-large-manifesto-cap', trust_remote_code=True); AutoTokenizer.from_pretrained('xlm-roberta-large')"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1', trust_remote_code=True); AutoTokenizer.from_pretrained('xlm-roberta-large')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/tweet-topic-latest-single')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='chkla/parlbert-topic-german')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='classla/xlm-roberta-base-multilingual-text-genre-classifier')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ssharoff/genres')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='classla/multilingual-IPTC-news-topic-classifier')"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('poltextlab/xlm-roberta-large-english-cap-v3', trust_remote_code=True); AutoTokenizer.from_pretrained('xlm-roberta-large')"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('poltextlab/xlm-roberta-large-party-cap-v3', trust_remote_code=True); AutoTokenizer.from_pretrained('xlm-roberta-large')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/tweet-topic-latest-single')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/tweet-topic-large-multilingual')"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('WebOrganizer/TopicClassifier', trust_remote_code=True, use_memory_efficient_attention=False); AutoTokenizer.from_pretrained('WebOrganizer/TopicClassifier')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='nickmuchi/finbert-tone-finetuned-finance-topic-classification')"
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




ENTRYPOINT ["uvicorn", "duui_transformers_topic:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
