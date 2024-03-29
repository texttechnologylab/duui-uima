FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -c "from transformers import pipeline; pipeline('text-classification', model='EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='FredZhang7/one-for-all-toxicity-v3')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='citizenlab/distilbert-base-multilingual-cased-toxicity')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='tomh/toxigen_hatebert')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='GroNLP/hateBERT')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='pysentimiento/bertweet-hate-speech')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/bert-base-uncased-hatexplain')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-hate-latest')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-german')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='deepset/bert-base-german-cased-hatespeech-GermEval18Coarse')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='martin-ha/toxic-comment-model')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='nicholasKluge/ToxicityModel')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='EIStakovskii/german_toxicity_classifier_plus_v2')"


# log level
ARG TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_LOG_LEVEL=$TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_LOG_LEVEL

# config
ARG TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_MODEL_CACHE_SIZE=1
ENV TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_MODEL_CACHE_SIZE=$TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_MODEL_CACHE_SIZE

# meta data
ARG TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_ANNOTATOR_NAME="textimager-duui-transformers-toxic"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_ANNOTATOR_NAME=$TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_ANNOTATOR_NAME
ARG TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_ANNOTATOR_VERSION=$TEXTIMAGER_DUUI_TRANSFORMERS_TOXIC_ANNOTATOR_VERSION

# offline mode for huggingface
#ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE

# copy scripts
COPY ./src/main/resources/TypeSystemToxic.xml ./src/main/resources/TypeSystemToxic.xml
COPY ./src/main/python/__init__.py ./src/main/python/__init__.py
COPY ./src/main/python/duui/ ./src/main/python/duui/
COPY ./src/main/python/ToxicCheck.py ./src/main/python/ToxicCheck.py
COPY ./src/main/python/textimager_duui_transformers_toxic.py ./src/main/python/textimager_duui_transformers_toxic.py
COPY ./src/main/lua/textimager_duui_transformers_toxic.lua ./src/main/lua/textimager_duui_transformers_toxic.lua


ENTRYPOINT ["uvicorn", "src.main.python.textimager_duui_transformers_toxic:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
