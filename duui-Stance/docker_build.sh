#!/usr/bin/env bash
set -euo pipefail

export ANNOTATOR_CUDA=
#export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-stance
export ANNOTATOR_VERSION=0.0.2
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export CHATGPT_KEY=
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

export MODEL_NAME="mlburnham"
export MODEL_VERSION="4538315b9903f9821063023bebcf441cb8c53cdc"
export MODEL_URL="https://huggingface.co/mlburnham/deberta-v3-base-polistance-affect-v1.0"
export MODEL_LANG="en"

#export MODEL_NAME="kornosk"
#export MODEL_VERSION="36311a4ad7200ac54d3e3aff37daee69d6472888"
#export MODEL_URL="https://huggingface.co/kornosk/bert-election2020-twitter-stance-trump"
#export MODEL_LANG="en"
#
#export MODEL_NAME="gpt3.5"
#export MODEL_VERSION="gpt-3.5-turbo"
#export MODEL_URL="https://platform.openai.com/"
#export MODEL_LANG="MULTI"
#
#export MODEL_NAME="gpt4"
#export MODEL_VERSION="gpt-4"
#export MODEL_URL="https://platform.openai.com/"
#export MODEL_LANG="MULTI"

docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  --build-arg MODEL_CACHE_SIZE \
  --build-arg MODEL_NAME \
  --build-arg MODEL_VERSION \
  --build-arg MODEL_URL \
  --build-arg MODEL_LANG \
  --build-arg CHATGPT_KEY \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_NAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  -f src/main/docker/Dockerfile${ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_NAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_NAME}:latest${ANNOTATOR_CUDA}
