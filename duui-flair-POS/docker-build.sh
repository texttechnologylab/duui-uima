#!/bin/bash
export CUDA_VERSION_PATCH="11.7.1"
export PYTHON_VERSION="3.10"
export PYTORCH_VERSION="1.13.1"

docker build \
    --build-arg CUDA_VERSION_PATCH="${CUDA_VERSION_PATCH}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg PYTORCH_VERSION="${PYTORCH_VERSION}" \
    -t "docker.texttechnologylab.org/flair/pos:latest" \
    -f src/main/docker/Dockerfile .
docker push docker.texttechnologylab.org/flair/pos:latest
