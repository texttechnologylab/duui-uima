FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='FredZhang7/one-for-all-toxicity-v3')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='citizenlab/distilbert-base-multilingual-cased-toxicity')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='martin-ha/toxic-comment-model')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='EIStakovskii/german_toxicity_classifier_plus_v2')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='nicholasKluge/ToxicityModel')"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('textdetox/xlmr-large-toxicity-classifier-v2', trust_remote_code=True); AutoTokenizer.from_pretrained('textdetox/xlmr-large-toxicity-classifier-v2')"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('tomh/toxigen_hatebert'); AutoTokenizer.from_pretrained('tomh/toxigen_hatebert')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='tomh/toxigen_hatebert')"
#RUN python -c "from detoxify import Detoxify; Detoxify('multilingual')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='s-nlp/roberta_toxicity_classifier')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='garak-llm/roberta_toxicity_classifier')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='s-nlp/russian_toxicity_classifier')"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large');"
#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('malexandersalazar/xlm-roberta-large-binary-cls-toxicity');"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='sismetanin/rubert-toxic-pikabu-2ch')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='textdetox/glot500-toxicity-classifier')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='textdetox/bert-multilingual-toxicity-classifier')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='dardem/xlm-roberta-large-uk-toxicity')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Xuhui/ToxDect-roberta-large')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus')"

# copy scripts
COPY ./src/main/python/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/python/duui_transformers_toxic.py ./duui_transformers_toxic.py
COPY ./src/main/python/duui_toxic.lua ./duui_toxic.lua
COPY ./src/main/python/ToxicCheck.py ./ToxicCheck.py

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config
ARG MODEL_CACHE_SIZE=1
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE

# meta data
ARG ANNOTATOR_NAME="duui-transformers-toxic"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION
ARG MODEL_VERSION=""
ENV MODEL_VERSION=$MODEL_VERSION

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
ARG DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE





ENTRYPOINT ["uvicorn", "duui_transformers_toxic:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
