#!/usr/bin/env bash
set -euo pipefail

export annotator_name=duui-vader-sentiment
export annotator_version=0.0.1

#export DUUI_VADER_SENTIMENT_LOG_LEVEL=DEBUG
export log_level=INFO

export DOCKER_REGISTRY="docker.texttechnologylab.org/"

docker build \
  --build-arg annotator_name \
  --build-arg annotator_version \
  --build-arg log_level \
  -t ${DOCKER_REGISTRY}${annotator_name}:${annotator_version} \
  -f "src/main/docker/Dockerfile" \
  . --no-cache

docker tag \
  ${DOCKER_REGISTRY}${annotator_name}:${annotator_version} \
  ${DOCKER_REGISTRY}${annotator_name}:latest
