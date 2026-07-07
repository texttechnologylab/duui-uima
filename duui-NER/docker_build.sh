#!/usr/bin/env bash
set -euo pipefail

export ANNOTATOR_CUDA=
#export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-ner
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=1
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

# Optional GLiNER/GLiNER2 settings
export NER_LABELS="person,organization,location,date,event,product,taxon,other"
export THRESHOLD=0.5
export BATCH_SIZE=8

###---------------------------------------------------------------------
# GLiNER
# Passend dazu im Dockerfile aktivieren:
# RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained(model_id='urchade/gliner_multi-v2.1', map_location='cpu')"
#export MODEL_NAME="gliner"
#export MODEL_SPECNAME="gliner-multi-v2-1"
#export MODEL_VERSION="443d26d654e0324125a96bebd8e796c14ff2efe6"
#export MODEL_SOURCE="https://huggingface.co/urchade/gliner_multi-v2.1"
#export MODEL_LANG="Multi"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
# GLiNER2
# Passend dazu im Dockerfile aktivieren:
# RUN python -c "from gliner2 import GLiNER2; GLiNER2.from_pretrained('fastino/gliner2-multi-v1')"
#export MODEL_NAME="gliner2"
#export MODEL_SPECNAME="gliner2-multi-v1"
#export MODEL_VERSION="cc151f5b0ce4f7010c3ae8884527dd43dddf9d21"
#export MODEL_SOURCE="https://huggingface.co/fastino/gliner2-multi-v1"
#export MODEL_LANG="Multi"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
# RoBERTa multilingual NER
# Passend dazu im Dockerfile aktivieren:
# RUN python -c "from transformers import pipeline; pipeline('token-classification', model='julian-schelb/roberta-ner-multilingual', aggregation_strategy='simple')"
#export MODEL_NAME="roberta-ner-multilingual"
#export MODEL_SPECNAME="roberta-ner-multilingual"
#export MODEL_VERSION="d0a19147f3bb0065c8091459e3d35405ce9d48da"
#export MODEL_SOURCE="https://huggingface.co/julian-schelb/roberta-ner-multilingual"
#export MODEL_LANG="Multi"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
# WikiNEuRal multilingual NER
# Passend dazu im Dockerfile aktivieren:
# RUN python -c "from transformers import pipeline; pipeline('token-classification', model='Babelscape/wikineural-multilingual-ner', aggregation_strategy='simple')"
#export MODEL_NAME="wikineural-multilingual-ner"
#export MODEL_SPECNAME="wikineural-multilingual-ner"
#export MODEL_VERSION="bed6ee7a45d2827b6c90a4fd7983f0241ae0a5c1"
#export MODEL_SOURCE="https://huggingface.co/Babelscape/wikineural-multilingual-ner"
#export MODEL_LANG="Multi"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
# XLM-R NER 40 languages
# Passend dazu im Dockerfile aktivieren:
# RUN python -c "from transformers import pipeline; pipeline('token-classification', model='nbroad/jplu-xlm-r-ner-40-lang', aggregation_strategy='simple')"
#export MODEL_NAME="xlm-r-ner-40-lang"
#export MODEL_SPECNAME="xlm-r-ner-40-lang"
#export MODEL_VERSION="7f7f0fe9bc946a9848611aff079f556387687216"
#export MODEL_SOURCE="https://huggingface.co/nbroad/jplu-xlm-r-ner-40-lang"
#export MODEL_LANG="Multi"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
export MODEL_NAME="flair-ner-german"
export MODEL_SPECNAME="flair-ner-german"
export MODEL_VERSION="4e3f3d15ba39ce3e00575a7a1de5da0ce8198ce7"
export MODEL_SOURCE="https://huggingface.co/flair/ner-german"
export MODEL_LANG="DE"
###--------------------------------------------------------------------


docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  --build-arg MODEL_CACHE_SIZE \
  --build-arg MODEL_NAME \
  --build-arg MODEL_VERSION \
  --build-arg MODEL_SOURCE \
  --build-arg MODEL_LANG \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  -f src/main/docker/Dockerfile${ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:latest${ANNOTATOR_CUDA}