#!/usr/bin/env bash
set -euo pipefail

export ANNOTATOR_CUDA=
#export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-transformers-emotion
export ANNOTATOR_VERSION=0.3.0
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

####---------------------------------------------------------------------
#export MODEL_NAME="02shanky/finetuned-twitter-xlm-roberta-base-emotion"
#export MODEL_SPECNAME="finetuned-twitter-xlm-roberta-base-emotion"
#export MODEL_VERSION="28e6d080e9f73171b574dd88ac768da9e6622c36"
#export MODEL_SOURCE="https://huggingface.co/02shanky/finetuned-twitter-xlm-roberta-base-emotion"
#export MODEL_LANG="Multi"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence"
#export MODEL_SPECNAME="dreamy-xlm-roberta-emotion"
#export MODEL_VERSION="b3487623ec2dd4b9bd0644d8266291afb9956e9f"
#export MODEL_SOURCE="https://huggingface.co/DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence"
#export MODEL_LANG="Multi"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="MilaNLProc/xlm-emo-t"
#export MODEL_SPECNAME="xlm-emo-t"
#export MODEL_VERSION="a6ee7c9fad08d60204e7ae437d41d392381496f0"
#export MODEL_SOURCE="https://huggingface.co/MilaNLProc/xlm-emo-t"
#export MODEL_LANG="Multi"
####--------------------------------------------------------------------


####---------------------------------------------------------------------
#export MODEL_NAME="j-hartmann/emotion-english-distilroberta-base"
#export MODEL_SPECNAME="emotion-english-distilroberta-base"
#export MODEL_VERSION="0e1cd914e3d46199ed785853e12b57304e04178b"
#export MODEL_SOURCE="https://huggingface.co/j-hartmann/emotion-english-distilroberta-base"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------


####---------------------------------------------------------------------
#export MODEL_NAME="michellejieli/emotion_text_classifier"
#export MODEL_SPECNAME="emotion_text_classifier"
#export MODEL_VERSION="dc4df5597fcda82589511c3900fedbe1c0ffec82"
#export MODEL_SOURCE="https://huggingface.co/michellejieli/emotion_text_classifier"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="cardiffnlp/twitter-roberta-base-emotion"
#export MODEL_SPECNAME="cardiffnlp-twitter-roberta-base-emotion"
#export MODEL_VERSION="2848306ad936b7cd47c76c2c4e14d694a41e0f54"
#export MODEL_SOURCE="https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="finiteautomata/bertweet-base-emotion-analysis"
#export MODEL_SPECNAME="bertweet-base-emotion-analysis"
#export MODEL_VERSION="c482c9e1750a29dcc393234816bcf468ff77cd2d	"
#export MODEL_SOURCE="https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="ActivationAI/distilbert-base-uncased-finetuned-emotion"
#export MODEL_SPECNAME="distilbert-base-uncased-finetuned-emotion"
#export MODEL_VERSION="dbf4470880ff3b73f22975241cd309bdf8e2195f"
#export MODEL_SOURCE="https://huggingface.co/ActivationAI/distilbert-base-uncased-finetuned-emotion"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="SamLowe/roberta-base-go_emotions"
#export MODEL_SPECNAME="roberta-base-go-emotions"
#export MODEL_VERSION="58b6c5b44a7a12093f782442969019c7e2982299"
#export MODEL_SOURCE="https://huggingface.co/SamLowe/roberta-base-go_emotions"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="mrm8488/t5-base-finetuned-emotion"
#export MODEL_SPECNAME="t5-base-finetuned-emotion"
#export MODEL_VERSION="e44a316825f11230724b36412fbf1899c76e82de"
#export MODEL_SOURCE="https://huggingface.co/mrm8488/t5-base-finetuned-emotion"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------


####---------------------------------------------------------------------
#export MODEL_NAME="pysentimiento"
#export MODEL_SPECNAME="pysentimiento"
#export MODEL_VERSION="60822acfd805ad5d95437c695daa33c18dbda060"
#export MODEL_SOURCE="https://github.com/pysentimiento/pysentimiento/"
#export MODEL_LANG="EN, ES, IT, PT"
####--------------------------------------------------------------------


####---------------------------------------------------------------------
#export MODEL_NAME="EmoAtlas"
#export MODEL_SPECNAME="emoatlas"
#export MODEL_VERSION="adae44a80dd55c1d1c467c4e72bdb2d8cf63bf28"
#export MODEL_SOURCE="https://github.com/alfonsosemeraro/emoatlas"
#export MODEL_LANG="EN"
####--------------------------------------------------------------------

###---------------------------------------------------------------------
export MODEL_NAME="pol_emo_mDeBERTa"
export MODEL_SPECNAME="pol_emo_mdeberta"
export MODEL_VERSION="523da7dc2523631787ef0712bad53bfe2ac46840"
export MODEL_SOURCE="https://github.com/tweedmann/pol_emo_mDeBERTa2"
export MODEL_LANG="Multi"
###--------------------------------------------------------------------


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