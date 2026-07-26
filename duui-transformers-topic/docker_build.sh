#!/usr/bin/env bash
set -euo pipefail

#export ANNOTATOR_CUDA=
export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-transformers-topic
export ANNOTATOR_VERSION=0.4.0
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"


###---------------------------------------------------------------------
#export MODEL_NAME="manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1"
#export MODEL_SPECNAME="manifestoberta-xlm-roberta"
#export MODEL_VERSION="06c046795a3b7b9822755f0a73776f8fabec3977"
#export MODEL_SOURCE="https://huggingface.co/manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="classla/multilingual-IPTC-news-topic-classifier"
#export MODEL_SPECNAME="multilingual-iptc-media-topic-classifier"
#export MODEL_VERSION="ad2fac9ca58ad554021c0f244f15a9d556976229"
#export MODEL_SOURCE="https://huggingface.co/classla/multilingual-IPTC-news-topic-classifier"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="poltextlab/xlm-roberta-large-english-cap-v3"
#export MODEL_SPECNAME="xlm-roberta-large-english-cap-v3"
#export MODEL_VERSION="580cb9cc334735b6cd09a8c2e050d19f5cebfeca"
#export MODEL_SOURCE="https://huggingface.co/poltextlab/xlm-roberta-large-english-cap-v3"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="poltextlab/xlm-roberta-large-party-cap-v3"
#export MODEL_SPECNAME="xlm-roberta-large-party-cap-v3"
#export MODEL_VERSION="42804267cb8db2cc056e96f9a6ceee01a579e126"
#export MODEL_SOURCE="https://huggingface.co/poltextlab/xlm-roberta-large-party-cap-v3"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="cardiffnlp/roberta-large-tweet-topic-single-all"
#export MODEL_SPECNAME="cardiffnlp-roberta-large-tweet-topic-single-all"
#export MODEL_VERSION="b9286fabc508a553a4dad6cec8035044deff034a"
#export MODEL_SOURCE="https://huggingface.co/cardiffnlp/roberta-large-tweet-topic-single-all"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="cardiffnlp/tweet-topic-large-multilingual"
#export MODEL_SPECNAME="tweet-topic-large-multilingual"
#export MODEL_VERSION="e68d741bf72c67d78806cf49a1f8831ffebd63f8"
#export MODEL_SOURCE="https://huggingface.co/cardiffnlp/tweet-topic-large-multilingual"
#export MODEL_LANG="EN,ES,El,JA"
##--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="WebOrganizer/TopicClassifier"
#export MODEL_SPECNAME="organize-web"
#export MODEL_VERSION="8d158c9d514cdc21a7c8e9bd94e5dc483d49e024"
#export MODEL_SOURCE="https://huggingface.co/WebOrganizer/TopicClassifier"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="classla/ParlaCAP-Topic-Classifier"
#export MODEL_SPECNAME="parlacap-topic-classifier"
#export MODEL_VERSION="bf5c7145d4266b4851063f458eaa5ba5e28a2c43"
#export MODEL_SOURCE="https://huggingface.co/classla/ParlaCAP-Topic-Classifier"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="yiyanghkust/finbert-esg-9-categories"
#export MODEL_SPECNAME="finbert-esg-9-categories"
#export MODEL_VERSION="af56509508a62691ad52c7a2d67798a6680502e7"
#export MODEL_SOURCE="https://huggingface.co/yiyanghkust/finbert-esg-9-categories"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="valurank/distilroberta-topic-classification"
#export MODEL_SPECNAME="distilroberta-topic-classification"
#export MODEL_VERSION="7699ea4103e8b5437bf6479365353cc972eb1ab0"
#export MODEL_SOURCE="https://huggingface.co/valurank/distilroberta-topic-classification"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="dstefa/roberta-base_topic_classification_nyt_news"
#export MODEL_SPECNAME="nyt-news-topic-classification"
#export MODEL_VERSION="3102e25f935cbcad5f9a81305f6c74218d93fc6a"
#export MODEL_SOURCE="https://huggingface.co/dstefa/roberta-base_topic_classification_nyt_news"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"
#export MODEL_SPECNAME="openalex-topic-classification"
#export MODEL_VERSION="3b352795992e06feed29639581fd34c922bc42f1"
#export MODEL_SOURCE="https://huggingface.co/OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="nickmuchi/finbert-tone-finetuned-finance-topic-classification"
#export MODEL_SPECNAME="finbert-tone-finance-topic-classification"
#export MODEL_VERSION="ee9b951e726648dba828e6b2b7035ddb4ff41759"
#export MODEL_SOURCE="https://huggingface.co/nickmuchi/finbert-tone-finetuned-finance-topic-classification"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="classla/ParlaCAP-Topic-Classifier"
export MODEL_SPECNAME="parlacap-topic-classifier"
export MODEL_VERSION="bf5c7145d4266b4851063f458eaa5ba5e28a2c43"
export MODEL_SOURCE="https://huggingface.co/classla/ParlaCAP-Topic-Classifier"
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
