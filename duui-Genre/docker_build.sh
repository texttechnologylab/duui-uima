#!/usr/bin/env bash
set -euo pipefail

export ANNOTATOR_CUDA=
#export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-genre
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

###---------------------------------------------------------------------
#export MODEL_NAME="TurkuNLP/web-register-classification-multilingual"
#export MODEL_SPECNAME="turkunlp-genre-multi"
#export MODEL_VERSION="a22ad8b652f6825ec1505dab779979e0f255d7ae"
#export MODEL_SOURCE="https://huggingface.co/TurkuNLP/web-register-classification-multilingual"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="TurkuNLP/web-register-classification-en"
#export MODEL_SPECNAME="turkunlp-genre-en"
#export MODEL_VERSION="93969151434144dc8505865d31823c79bd385167"
#export MODEL_SOURCE="https://huggingface.co/TurkuNLP/web-register-classification-en"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="TurkuNLP/finerweb-quality-classifier"
#export MODEL_SPECNAME="turkunlp-genre-finerweb"
#export MODEL_VERSION="93d1635105c974a675e3be8c636d7a5cac6f7b11"
#export MODEL_SOURCE="https://huggingface.co/TurkuNLP/finerweb-quality-classifier"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="ssharoff/genres"
#export MODEL_SPECNAME="ssharoff-genre"
#export MODEL_VERSION="93d1635105c974a675e3be8c636d7a5cac6f7b11"
#export MODEL_SOURCE="https://huggingface.co/ssharoff/genres"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="classla/xlm-roberta-base-multilingual-text-genre-classifier"
export MODEL_SPECNAME="x-genre-classifier"
export MODEL_VERSION="ebe54ca322f6fd4dc95700705b99f23e3437c8d0"
export MODEL_SOURCE="https://huggingface.co/classla/xlm-roberta-base-multilingual-text-genre-classifier"
export MODEL_LANG="Multi"
##--------------------------------------------------------------------



docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  --build-arg MODEL_CACHE_SIZE \
  --build-arg MODEL_NAME \
  --build-arg MODEL_VERSION \
  --build-arg MODEL_SOURCE \
  --build-arg MODEL_LANG \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  -f src/main/docker/Dockerfile${ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:latest${ANNOTATOR_CUDA}
