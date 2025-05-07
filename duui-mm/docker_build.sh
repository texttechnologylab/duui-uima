#!/usr/bin/env bash
set -euo pipefail


export MM_ANNOTATOR_CUDA=
#export DUUI_MM_CUDA="-cuda"

export MM_ANNOTATOR_NAME=duui-mutlimodality
export MM_ANNOTATOR_VERSION=0.1.0
export MM_LOG_LEVEL=DEBUG
export MM_MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"


docker build \
  --build-arg MM_ANNOTATOR_NAME \
  --build-arg MM_ANNOTATOR_VERSION \
  --build-arg MM_LOG_LEVEL \
  -t ${DOCKER_REGISTRY}${MM_ANNOTATOR_NAME}:${MM_ANNOTATOR_VERSION}${MM_ANNOTATOR_CUDA} \
  -f src/main/docker/Dockerfile${MM_ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${MM_ANNOTATOR_NAME}:${MM_ANNOTATOR_VERSION}${MM_ANNOTATOR_CUDA} \
  ${DOCKER_REGISTRY}${MM_ANNOTATOR_NAME}:latest${MM_ANNOTATOR_CUDA}
