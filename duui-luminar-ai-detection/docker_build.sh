#!/usr/bin/env bash
set -euo pipefail

export DUUI_ANNOTATOR_NAME=luminar-seq
export DUUI_ANNOTATOR_VERSION=1.0.0

export TTLAB_DUUI_DOCKER_REGISTRY="docker.texttechnologylab.org/"

# Build the Docker image
docker build \
  --build-arg DUUI_ANNOTATOR_NAME \
  --build-arg DUUI_ANNOTATOR_VERSION \
  -t ${TTLAB_DUUI_DOCKER_REGISTRY}${DUUI_ANNOTATOR_NAME}:${DUUI_ANNOTATOR_VERSION} \
  -f "src/main/docker/Dockerfile" \
  .

# Automatically tag the newest image as "latest"
docker tag \
  ${TTLAB_DUUI_DOCKER_REGISTRY}${DUUI_ANNOTATOR_NAME}:${DUUI_ANNOTATOR_VERSION} \
  ${TTLAB_DUUI_DOCKER_REGISTRY}${DUUI_ANNOTATOR_NAME}:latest