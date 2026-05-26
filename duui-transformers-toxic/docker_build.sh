#!/usr/bin/env bash
set -euo pipefail

export ANNOTATOR_CUDA=
export DUUI_CUDA=
#export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME=duui-transformers-toxic
export ANNOTATOR_VERSION=0.4.0
#export MODEL_VERSION="0.4.0"
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=3

##---------------------------------------------------------------------
#export MODEL_NAME="EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus"
#export MODEL_SPECNAME="multilingual_toxicity_classifier_plus"
#export MODEL_VERSION="0126552291025f2fc854f5acdbe45b2212eabf4a"
#export MODEL_SOURCE="https://huggingface.co/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus"
#export MODEL_LANG="Multi"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="FredZhang7/one-for-all-toxicity-v3"
#export MODEL_SPECNAME="one_for_all_toxicity_v3"
#export MODEL_VERSION="a2996bd4495269071eaf5daf73512234c33cb3d2"
#export MODEL_SOURCE="https://huggingface.co/FredZhang7/one-for-all-toxicity-v3"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="citizenlab/distilbert-base-multilingual-cased-toxicity"
#export MODEL_SPECNAME="distilbert_base_multilingual_cased_toxicity"
#export MODEL_VERSION="b4532a8b095d1886a7b5dff818331ecc88a855ae"
#export MODEL_SOURCE="https://huggingface.co/citizenlab/distilbert-base-multilingual-cased-toxicity"
#export MODEL_LANG="EN, FR, NL, PT, IT, SP, DE, PL, DA, AF"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="martin-ha/toxic-comment-model"
#export MODEL_SPECNAME="toxic_comment_model"
#export MODEL_VERSION="9842c08b35a4687e7b211187d676986c8c96256d"
#export MODEL_SOURCE="https://huggingface.co/martin-ha/toxic-comment-model"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="EIStakovskii/german_toxicity_classifier_plus_v2"
#export MODEL_SPECNAME="german_toxicity_classifier_plus_v2"
#export MODEL_VERSION="1bcb7d11ffc9267111c7be1dad0d7ca2fbf73928"
#export MODEL_SOURCE="https://huggingface.co/EIStakovskii/german_toxicity_classifier_plus_v2"
#export MODEL_LANG="DE"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="nicholasKluge/ToxicityModel"
#export MODEL_SPECNAME="aira_toxicity_model"
#export MODEL_VERSION="900a6eab23ddd93f6c282f1752eb1fb5e9879d86"
#export MODEL_SOURCE="https://huggingface.co/nicholasKluge/ToxicityModel"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="nicholasKluge/ToxicityModel"
#export MODEL_SPECNAME="aira_toxicity_model"
#export MODEL_VERSION="900a6eab23ddd93f6c282f1752eb1fb5e9879d86"
#export MODEL_SOURCE="https://huggingface.co/nicholasKluge/ToxicityModel"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="textdetox/xlmr-large-toxicity-classifier"
#export MODEL_SPECNAME="xlmr_large_toxicity_classifier"
#export MODEL_VERSION="b9c7c563427c591fc318d91eb592381ae2fbde66"
#export MODEL_SOURCE="https://huggingface.co/textdetox/xlmr-large-toxicity-classifier"
#export MODEL_LANG="Multi"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="tomh/toxigen_roberta"
#export MODEL_SPECNAME="toxigen"
#export MODEL_VERSION="0e65216a558feba4bb167d47e49f9a9e229de6ab"
#export MODEL_SOURCE="https://huggingface.co/tomh/toxigen_roberta"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------


##---------------------------------------------------------------------
#export MODEL_NAME="Detoxify"
#export MODEL_SPECNAME="detoxify"
#export MODEL_VERSION="8f56f302bf8cf2673c2132fc2c2f5b2ca804815f"
#export MODEL_SOURCE="https://github.com/unitaryai/detoxify"
#export MODEL_LANG="EN, FR, ES, IT, PT, TR, RU"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="s-nlp/roberta_toxicity_classifier"
#export MODEL_SPECNAME="roberta_toxicity_classifier"
#export MODEL_VERSION="048c25bb1e199b98802784f96325f4840f22145d"
#export MODEL_SOURCE="https://huggingface.co/s-nlp/roberta_toxicity_classifier"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="garak-llm/roberta_toxicity_classifier"
#export MODEL_SPECNAME="para-detox"
#export MODEL_VERSION="fb7e9d615fc8c59d2e70466a831ed70d6f6f895a"
#export MODEL_SOURCE="https://huggingface.co/garak-llm/roberta_toxicity_classifier"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="s-nlp/russian_toxicity_classifier"
#export MODEL_SPECNAME="russe-2022"
#export MODEL_VERSION="0694e1f99efc08e73479e5c6f06c7bbe393aca89"
#export MODEL_SOURCE="https://huggingface.co/s-nlp/russian_toxicity_classifier"
#export MODEL_LANG="RU"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="malexandersalazar/xlm-roberta-large-binary-cls-toxicity"
#export MODEL_SPECNAME="xlm-multi-toxic"
#export MODEL_VERSION="6968ce7aa290a1bb2bbada047a3491aa048e2bd3"
#export MODEL_SOURCE="https://huggingface.co/malexandersalazar/xlm-roberta-large-binary-cls-toxicity"
#export MODEL_LANG="EN,DE,FR,IT,PT,TH,HI,ES"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="sismetanin/rubert-toxic-pikabu-2ch"
#export MODEL_SPECNAME="rubert-toxic"
#export MODEL_VERSION="1e5d55aeca25ab0a91725abc08821694de7dd5ea"
#export MODEL_SOURCE="https://huggingface.co/sismetanin/rubert-toxic-pikabu-2ch"
#export MODEL_LANG="RU"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="textdetox/glot500-toxicity-classifier"
#export MODEL_SPECNAME="textdetox-glot500"
#export MODEL_VERSION="4c2e8b298c4c7980d23566e92ab68b53f30db025"
#export MODEL_SOURCE="https://huggingface.co/textdetox/glot500-toxicity-classifier"
#export MODEL_LANG="EN,FR,IT,ES,RU,UK,AR,HI,JA,ZH,DE,TT,HE,AM"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="textdetox/bert-multilingual-toxicity-classifier"
#export MODEL_SPECNAME="textdetox-bert"
#export MODEL_VERSION="0667d0fbb85a1ea7b1e3a1f2a9a2901f5ce8c16c"
#export MODEL_SOURCE="https://huggingface.co/textdetox/bert-multilingual-toxicity-classifier"
#export MODEL_LANG="EN,FR,IT,ES,RU,UK,AR,HI,JA,ZH,DE,TT,HE,AM"
##--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="dardem/xlm-roberta-large-uk-toxicity"
#export MODEL_SPECNAME="toxicity-classifier-uk"
#export MODEL_VERSION="6e2c8c305cc7ccff14a6dfe3d8fdd83d6556f514"
#export MODEL_SOURCE="https://huggingface.co/dardem/xlm-roberta-large-uk-toxicity"
#export MODEL_LANG="UK"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="Xuhui/ToxDect-roberta-large"
#export MODEL_SPECNAME="toxdect"
#export MODEL_VERSION="7b97c89938cb241d3ae9235257bbe4916d4f0c75"
#export MODEL_SOURCE="https://huggingface.co/Xuhui/ToxDect-roberta-large"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus"
export MODEL_SPECNAME="multi-toxic-classifier-plus"
export MODEL_VERSION="0126552291025f2fc854f5acdbe45b2212eabf4a"
export MODEL_SOURCE="https://huggingface.co/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus"
export MODEL_LANG="EN,RU,FR,DE"
##--------------------------------------------------------------------


export DOCKER_REGISTRY="docker.texttechnologylab.org/"


docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  --build-arg MODEL_CACHE_SIZE \
  --build-arg MODEL_VERSION \
  --build-arg MODEL_NAME \
  --build-arg MODEL_VERSION \
  --build-arg MODEL_SOURCE \
  --build-arg MODEL_LANG \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  -f src/main/docker/Dockerfile${DUUI_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:latest${DUUI_CUDA}
