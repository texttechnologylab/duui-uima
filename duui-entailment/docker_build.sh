#!/usr/bin/env bash
set -euo pipefail

export ANNOTATOR_CUDA=
#export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-entailment
export ANNOTATOR_VERSION=0.0.1
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export CHATGPT_KEY=
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

#export MODEL_NAME="google/flan-t5-base"
#export SHORT_MODEL_NAME="google-flan-t5-base"
#export MODEL_VERSION="7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
#export MODEL_URL="https://huggingface.co/google/flan-t5-base"
#export MODEL_LANG="Multilingual"

#export MODEL_NAME="google/flan-t5-small"
#export SHORT_MODEL_NAME="google-flan-t5-small"
#export MODEL_VERSION="0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
#export MODEL_URL="https://huggingface.co/google/flan-t5-small"
#export MODEL_LANG="Multilingual"

#export MODEL_NAME="google/flan-t5-large"
#export SHORT_MODEL_NAME="google-flan-t5-large"
#export MODEL_VERSION="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
#export MODEL_URL="https://huggingface.co/google/flan-t5-large"
#export MODEL_LANG="Multilingual"

#export MODEL_NAME="google/flan-t5-xl"
#export SHORT_MODEL_NAME="google-flan-t5-xl"
#export MODEL_VERSION="7d6315df2c2fb742f0f5b556879d730926ca9001"
#export MODEL_URL="https://huggingface.co/google/flan-t5-xl"
#export MODEL_LANG="Multilingual"

#export MODEL_NAME="google/flan-t5-xxl"
#export SHORT_MODEL_NAME="google-flan-t5-xxl"
#export MODEL_VERSION="ae7c9136adc7555eeccc78cdd960dfd60fb346ce"
#export MODEL_URL="https://huggingface.co/google/flan-t5-xxl"
#export MODEL_LANG="Multilingual"

export MODEL_NAME="soumyasanyal/entailment-verifier-xxl"
export SHORT_MODEL_NAME="soumyasanyal-entailment-verifier-xxl"
export MODEL_VERSION="f11b55bc0304bf0b2fa08eb5311cd26dacca482a"
export MODEL_URL="https://huggingface.co/soumyasanyal/entailment-verifier-xxl"
export MODEL_LANG="en"
#
#export MODEL_NAME="gpt3.5"
#export SHORT_MODEL_NAME="gpt3.5"
#export MODEL_VERSION="gpt-3.5-turbo"
#export MODEL_URL="https://platform.openai.com/"
#export MODEL_LANG="MULTI"

#export MODEL_NAME="gpt4"
#export SHORT_MODEL_NAME="gpt4"
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
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${SHORT_MODEL_NAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  -f src/main/docker/Dockerfile${ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${SHORT_MODEL_NAME}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${SHORT_MODEL_NAME}:latest${ANNOTATOR_CUDA}
