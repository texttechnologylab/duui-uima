#!/bin/bash

export ANNOTATOR_NAME=duui-youtube-downloader
export ANNOTATOR_VERSION=0.1
export LOG_LEVEL=DEBUG
export DOCKER_REGISTRY="docker.texttechnologylab.org/"


docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:${ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile \
  .
 #--no-cache

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:${ANNOTATOR_VERSION} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}:latest

