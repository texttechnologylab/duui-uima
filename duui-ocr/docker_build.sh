#!/usr/bin/env bash
set -euo pipefail

export DUUI_OCR_CUDA=
#export DUUI_OCR_CUDA="-cuda"

export DUUI_OCR_ANNOTATOR_NAME=duui-ocr
export DUUI_OCR_ANNOTATOR_VERSION=0.2.0
export DUUI_OCR_LOG_LEVEL=DEBUG
export DUUI_OCR_MODEL_CACHE_SIZE=1
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

docker build \
  --build-arg DUUI_OCR_ANNOTATOR_NAME \
  --build-arg DUUI_OCR_ANNOTATOR_VERSION \
  --build-arg DUUI_OCR_LOG_LEVEL \
  --build-arg DUUI_OCR_MODEL_CACHE_SIZE \
  -t ${DOCKER_REGISTRY}${DUUI_OCR_ANNOTATOR_NAME}:${DUUI_OCR_ANNOTATOR_VERSION}${DUUI_OCR_CUDA} \
  -f src/main/docker/Dockerfile${DUUI_OCR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${DUUI_OCR_ANNOTATOR_NAME}:${DUUI_OCR_ANNOTATOR_VERSION}${DUUI_OCR_CUDA} \
  ${DOCKER_REGISTRY}${DUUI_OCR_ANNOTATOR_NAME}:latest${DUUI_OCR_CUDA}