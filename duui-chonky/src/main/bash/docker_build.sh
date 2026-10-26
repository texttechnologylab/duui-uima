#!/usr/bin/env bash
set -euo pipefail

export DUUI_CHONKY_ANNOTATOR_NAME=duui-chonky
export DUUI_CHONKY_ANNOTATOR_VERSION=1.0.0

export DUUI_CHONKY_LOG_LEVEL=INFO

export DOCKER_REGISTRY="docker.texttechnologylab.org/"

docker build \
  --build-arg DUUI_CHONKY_ANNOTATOR_NAME \
  --build-arg DUUI_CHONKY_ANNOTATOR_VERSION \
  --build-arg DUUI_CHONKY_LOG_LEVEL \
  -t ${DOCKER_REGISTRY}${DUUI_CHONKY_ANNOTATOR_NAME}:${DUUI_CHONKY_ANNOTATOR_VERSION} \
  -f "src/main/docker/Dockerfile" \
  .

docker tag \
  ${DOCKER_REGISTRY}${DUUI_CHONKY_ANNOTATOR_NAME}:${DUUI_CHONKY_ANNOTATOR_VERSION} \
  ${DOCKER_REGISTRY}${DUUI_CHONKY_ANNOTATOR_NAME}:latest
