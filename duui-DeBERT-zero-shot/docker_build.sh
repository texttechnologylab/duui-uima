#!/usr/bin/env bash
set -euo pipefail

export ANNOTATOR_NAME=debert-zero-shot
export ANNOTATOR_VERSION=0.1

export LOG_LEVEL=INFO

export DOCKER_REGISTRY="docker.texttechnologylab.org/"

docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:${ANNOTATOR_VERSION} \
  -f "./Dockerfile" \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:${ANNOTATOR_VERSION} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:latest
