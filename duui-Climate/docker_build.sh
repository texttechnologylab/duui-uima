#!/usr/bin/env bash
set -euo pipefail

export ANNOTATOR_CUDA=
#export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-climate
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

###---------------------------------------------------------------------
#export MODEL_NAME="climatebert/distilroberta-base-climate-detector"
#export MODEL_SPECNAME="distilroberta-base-climate-detector"
#export MODEL_VERSION="2c3bc660d45a59e31b35f5d3e365ee4f59fdf76c"
#export MODEL_SOURCE="https://huggingface.co/climatebert/distilroberta-base-climate-detector"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="climatebert/distilroberta-base-climate-tcfd"
#export MODEL_SPECNAME="distilroberta-base-climate-tcfd"
#export MODEL_VERSION="970630beedc21db81a84156448ad2e3ac860153d"
#export MODEL_SOURCE="https://huggingface.co/climatebert/distilroberta-base-climate-tcfd"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="climatebert/distilroberta-base-climate-commitment"
#export MODEL_SPECNAME="distilroberta-base-climate-commitment"
#export MODEL_VERSION="17337c3292df16a8fe93b1505dfe4122d50a4c91"
#export MODEL_SOURCE="https://huggingface.co/climatebert/distilroberta-base-climate-commitment"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="climatebert/distilroberta-base-climate-sentiment"
#export MODEL_SPECNAME="distilroberta-base-climate-sentiment"
#export MODEL_VERSION="e9f9a94ee4263f5ad5cfc97b8539a497fc88aa7d"
#export MODEL_SOURCE="https://huggingface.co/climatebert/distilroberta-base-climate-sentiment"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="climatebert/distilroberta-base-climate-specificity"
export MODEL_SPECNAME="distilroberta-base-climate-specificity"
export MODEL_VERSION="4ada96ed4bf5c3a7a711282e41f1ab9b29f0ddea"
export MODEL_SOURCE="https://huggingface.co/climatebert/distilroberta-base-climate-specificity"
export MODEL_LANG="EN"
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
