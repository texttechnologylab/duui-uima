#!/usr/bin/env bash
set -euo pipefail

#export ANNOTATOR_CUDA=
export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-transformers-toxic
export ANNOTATOR_VERSION=0.3.0
export MODEL_VERSION="0.3.0"
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"


docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  --build-arg MODEL_CACHE_SIZE \
  --build-arg MODEL_VERSION \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  -f src/main/docker/Dockerfile${ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:latest${ANNOTATOR_CUDA}
