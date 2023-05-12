#!/bin/bash

export TEXTIMAGER_COREF_ANNOTATOR_NAME=gercoref
export TEXTIMAGER_COREF_ANNOTATOR_VERSION=0.0.6
export TEXTIMAGER_COREF_LOG_LEVEL=DEBUG
export TEXTIMAGER_COREF_PARSER_MODEL_NAME=se10_electra_uncased

if [[ ! -f src/main/python/models/model_se10_electra_uncased_Apr12_16-08-17_42300.bin ]]
then
  wget https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiRwaDmKVL78HRtJzLE2jQ/model_se10_electra_uncased_Apr12_16-08-17_42300.bin -P src/main/python/models/
fi


docker build \
  --build-arg TEXTIMAGER_COREF_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_COREF_ANNOTATOR_VERSION \
  --build-arg TEXTIMAGER_COREF_LOG_LEVEL \
  --build-arg TEXTIMAGER_COREF_PARSER_MODEL_NAME \
  -t ${TEXTIMAGER_COREF_ANNOTATOR_NAME}:${TEXTIMAGER_COREF_ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile \
  .
