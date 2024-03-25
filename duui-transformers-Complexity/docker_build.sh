#!/usr/bin/env bash
set -euo pipefail

export COMPLEXITY_ANNOTATOR_CUDA=
#export COMPLEXITY_ANNOTATOR_CUDA="-cuda"

export COMPLEXITY_ANNOTATOR_NAME=duui-transformers-complexity
export COMPLEXITY_ANNOTATOR_VERSION=0.1.0
export COMPLEXITY_LOG_LEVEL=DEBUG
export COMPLEXITY_MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

#export COMPLEXITY_MODEL_NAME="intfloat/multilingual-e5-base"
#export COMPLEXITY_MODEL_VERSION="d13f1b27baf31030b7fd040960d60d909913633f"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/intfloat/multilingual-e5-base"
#export COMPLEXITY_MODEL_ART="BertSentence"
#export COMPLEXITY_MODEL_LANG="Multi"

#export COMPLEXITY_MODEL_NAME="google-bert/bert-base-multilingual-cased"
#export COMPLEXITY_MODEL_VERSION="3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/google-bert/bert-base-multilingual-cased"
#export COMPLEXITY_MODEL_ART="Bert"
#export COMPLEXITY_MODEL_LANG="Multi"

#export COMPLEXITY_MODEL_NAME="FacebookAI/xlm-roberta-large"
#export COMPLEXITY_MODEL_VERSION="c23d21b0620b635a76227c604d44e43a9f0ee389"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/FacebookAI/xlm-roberta-large"
#export COMPLEXITY_MODEL_ART="Bert"
#export COMPLEXITY_MODEL_LANG="Multi"

#export COMPLEXITY_MODEL_NAME="facebook/xlm-v-base"
#export COMPLEXITY_MODEL_VERSION="68c75dd7733d2640b3a98114e3e94196dc543fe1"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/facebook/xlm-v-base"
#export COMPLEXITY_MODEL_ART="Bert"
#export COMPLEXITY_MODEL_LANG="Multi"

#export COMPLEXITY_MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base"
#export COMPLEXITY_MODEL_VERSION="4c365f1490cb329b52150ad72f922ea467b5f4e6"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base"
#export COMPLEXITY_MODEL_ART="Bert"
#export COMPLEXITY_MODEL_LANG="Multi"

#export COMPLEXITY_MODEL_NAME="setu4993/LEALLA-small"
#export COMPLEXITY_MODEL_VERSION="8fadf81fe3979f373ba9922ab616468a4184b266"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/setu4993/LEALLA-small"
#export COMPLEXITY_MODEL_ART="BertSentence"
#export COMPLEXITY_MODEL_LANG="Multi"
#
#export COMPLEXITY_MODEL_NAME="sentence-transformers/LaBSE"
#export COMPLEXITY_MODEL_VERSION="5513ed8dd44a9878c7d4fe8646d4dd9df2836b7b"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/sentence-transformers/LaBSE"
#export COMPLEXITY_MODEL_ART="Sentence"
#export COMPLEXITY_MODEL_LANG="Multi"
#
#export COMPLEXITY_MODEL_NAME="Twitter/twhin-bert-large"
#export COMPLEXITY_MODEL_VERSION="2786782c0f659550e3492093e4aab963d495243"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/Twitter/twhin-bert-large"
#export COMPLEXITY_MODEL_ART="Bert"
#export COMPLEXITY_MODEL_LANG="Multi"
#
#export COMPLEXITY_MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#export COMPLEXITY_MODEL_VERSION="543dcf585e1eb6d4ece18c2e0c29474d9c5146b70"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#export COMPLEXITY_MODEL_ART="Sentence"
#export COMPLEXITY_MODEL_LANG="Multi"
#
#export COMPLEXITY_MODEL_NAME="sentence-transformers/distiluse-base-multilingual-cased-v2"
#export COMPLEXITY_MODEL_VERSION="501a2afbd9deb9f028b175cc6060f38bb5055ce4"
#export COMPLEXITY_MODEL_SOURCE="https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2"
#export COMPLEXITY_MODEL_ART="Sentence"
#export COMPLEXITY_MODEL_LANG="Multi"


docker build \
  --build-arg COMPLEXITY_ANNOTATOR_NAME \
  --build-arg COMPLEXITY_ANNOTATOR_VERSION \
  --build-arg COMPLEXITY_LOG_LEVEL \
  -t ${DOCKER_REGISTRY}${COMPLEXITY_ANNOTATOR_NAME}:${COMPLEXITY_ANNOTATOR_VERSION}${COMPLEXITY_ANNOTATOR_CUDA} \
  -f src/main/docker/Dockerfile${COMPLEXITY_ANNOTATOR_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${COMPLEXITY_ANNOTATOR_NAME}:${COMPLEXITY_ANNOTATOR_VERSION}${COMPLEXITY_ANNOTATOR_CUDA} \
  ${DOCKER_REGISTRY}${COMPLEXITY_ANNOTATOR_NAME}:latest${COMPLEXITY_ANNOTATOR_CUDA}
