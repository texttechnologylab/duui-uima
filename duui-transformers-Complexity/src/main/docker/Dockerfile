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

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='02shanky/finetuned-twitter-xlm-roberta-base-emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='MilaNLProc/xlm-emo-t')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='michellejieli/emotion_text_classifier')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='finiteautomata/bertweet-base-emotion-analysis')"

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
ARG COMPLEXITY_MODEL_VERSION=0.1
ENV COMPLEXITY_MODEL_VERSION=$COMPLEXITY_MODEL_VERSION

# offline mode for huggingface
#ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE




ENTRYPOINT ["uvicorn", "duui_transformers_complexity:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
