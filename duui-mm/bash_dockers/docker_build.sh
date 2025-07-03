#!/usr/bin/env bash
set -euo pipefail


export MM_ANNOTATOR_CUDA=transformer
#export DUUI_MM_CUDA="-cuda"

export MM_ANNOTATOR_NAME=duui-mutlimodality
export MM_ANNOTATOR_VERSION=0.2.0
export MM_LOG_LEVEL=DEBUG
export MM_MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

cd ..

docker build \
  --build-arg MM_ANNOTATOR_NAME \
  --build-arg MM_ANNOTATOR_VERSION \
  --build-arg MM_LOG_LEVEL \
  -t ${DOCKER_REGISTRY}${MM_ANNOTATOR_NAME}-${MM_ANNOTATOR_CUDA}:${MM_ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile${MM_ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${MM_ANNOTATOR_NAME}-${MM_ANNOTATOR_CUDA}:${MM_ANNOTATOR_VERSION} \
  ${DOCKER_REGISTRY}${MM_ANNOTATOR_NAME}-${MM_ANNOTATOR_CUDA}:latest
