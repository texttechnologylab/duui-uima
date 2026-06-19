FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      --timeout 300 \
      --retries 30 \
      --progress-bar off \
      -r requirements.txt

# MODEL_NAME=gliner
#RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained(model_id='urchade/gliner_multi-v2.1', map_location='cpu')"
# MODEL_NAME=gliner2
#RUN python -c "from gliner2 import GLiNER2; GLiNER2.from_pretrained('fastino/gliner2-multi-v1')"
# MODEL_NAME=roberta-ner-multilingual
#RUN python -c "from transformers import pipeline; pipeline('token-classification', model='julian-schelb/roberta-ner-multilingual', aggregation_strategy='simple')"
# MODEL_NAME=wikineural-multilingual-ner
#RUN python -c "from transformers import pipeline; pipeline('token-classification', model='Babelscape/wikineural-multilingual-ner', aggregation_strategy='simple')"
# MODEL_NAME=xlm-r-ner-40-lang
#RUN python -c "from transformers import pipeline; pipeline('token-classification', model='nbroad/jplu-xlm-r-ner-40-lang', aggregation_strategy='simple')"
# MODEL_NAME=flair-ner-german
RUN python -c "from flair.models import SequenceTagger; SequenceTagger.load('flair/ner-german')"

# copy DUUI NER scripts
COPY ./src/main/python/TypeSystemNER.xml ./TypeSystemNER.xml
COPY ./src/main/python/duui_ner.py ./duui_ner.py
COPY ./src/main/python/ner_classification_backend.py ./ner_classification_backend.py
COPY ./src/main/python/duui_ner.lua ./duui_ner.lua


# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config: one model per container
ARG MODEL_CACHE_SIZE=1
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE

# meta data
ARG ANNOTATOR_NAME="duui-ner"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# Model info
# Genau EIN Modell pro Container. Keine kommagetrennten Listen, kein "all".
ARG MODEL_NAME="wikineural-multilingual-ner"
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_VERSION="0.1"
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG="multi"
ENV MODEL_LANG=$MODEL_LANG

# offline mode for HuggingFace/runtime
ARG TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE
ENV HF_HUB_OFFLINE=$TRANSFORMERS_OFFLINE

ENTRYPOINT ["uvicorn", "duui_ner:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
