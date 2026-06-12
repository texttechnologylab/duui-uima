#!/usr/bin/env bash
set -euo pipefail

#export ANNOTATOR_CUDA=
export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-relation-extraction
export ANNOTATOR_VERSION=0.4.0
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

####---------------------------------------------------------------------
export MODEL_NAME="Babelscape/rebel-large"
export MODEL_SPECNAME="rebel-large"
export MODEL_VERSION="44eb6cb4585df284ce6c4d6a7013f83fe473c052"
export MODEL_SOURCE="https://huggingface.co/Babelscape/rebel-large"
export MODEL_LANG="Multilingual"
####--------------------------------------------------------------------

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


####---------------------------------------------------------------------
export MODEL_NAME="ibm-research/knowgl-large"
export MODEL_SPECNAME="knowgl-large"
export MODEL_VERSION="94596fd9f697498f7ee7363dbf4cc66f08d499e8"
export MODEL_SOURCE="https://huggingface.co/ibm-research/knowgl-large"
export MODEL_LANG="Multilingual"
####--------------------------------------------------------------------


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

