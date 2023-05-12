#!/usr/bin/env bash
set -euo pipefail

#export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_CUDA=
export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_CUDA="-cuda"

export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_VARIANT=
#export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_VARIANT="-adapter"

export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_NAME=textimager-duui-transformers-sentiment
export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_VERSION=0.1.2

#export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_LOG_LEVEL=DEBUG
export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_LOG_LEVEL=INFO

export TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_MODEL_CACHE_SIZE=3

export DOCKER_REGISTRY="docker.texttechnologylab.org/"

# mount external model dir to access in docker context
# TODO these models should better be accessed via a server...
#model_mount_dirs=()
#mount_model () {
#  local external_model
#  external_model="$(pwd)/models/$1"
#  echo "mounting $external_model"
#  mkdir -p "$external_model"
#  bindfs --no-allow-other "$2/$1" "$external_model"
#  model_mount_dirs+=( "$external_model" )
#}

#EXP_DE_3_MODELS_BASE_DIR="/mnt/corpora2/projects/baumartz/ma/data/models/experiments/de/3sentiment"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-35057" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-70114" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-105171" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-140228" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-175285" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-35057" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-70114" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-105171" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-140228" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment/checkpoint-175285" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-35057" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-70114" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-105171" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-140228" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment/checkpoint-175285" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-35057" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-70114" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-105171" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-140228" "$EXP_DE_3_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-175285" "$EXP_DE_3_MODELS_BASE_DIR/"

#EXP_DE_3_UNSEEN_MODELS_BASE_DIR="/mnt/corpora2/projects/baumartz/ma/data/models/experiments/de/3sentiment-unseen"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-35010" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-70020" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-105030" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-140040" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-175050" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-4193" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-8386" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-12579" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-16772" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-20965" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-4224" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-8448" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-12672" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-16896" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-21120" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-30870" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-61740" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-92610" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-123480" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-154350" "$EXP_DE_3_UNSEEN_MODELS_BASE_DIR/"

#EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR="/mnt/corpora2/projects/baumartz/ma/data/models/experiments/de/3sentiment-unseen"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-210060" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-420120" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-630180" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-840240" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-1050300" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-25156" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-50312" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-75468" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-100624" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-125780" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-25342" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-50684" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-76026" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-101368" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-126710" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-185216" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-370432" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-555648" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-740864" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-926080" "$EXP_DE_3_UNSEEN_ADAPTER_MODELS_BASE_DIR/"

#EXP_DE_3_EXACT_MODELS_BASE_DIR="/mnt/corpora2/projects/baumartz/ma/data/models/experiments/de/3sentiment-exact"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-30979" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-61958" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-92937" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-123916" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-154895" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-30979" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-61958" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-92937" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-123916" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-154895" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-30979" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-61958" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-92937" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-123916" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-154895" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-30979" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-61958" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-92937" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-123916" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-154895" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"

#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-185874" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-216853" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-247832" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-278811" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-309790" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-340769" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-371748" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-402727" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-433706" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-464685" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-495664" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-526643" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-557622" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-588601" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-619580" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"

#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-185874" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-216853" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-247832" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-278811" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-309790" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-340769" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-371748" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-402727" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-433706" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-464685" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-495664" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-526643" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-557622" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-588601" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-619580" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"

#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-185874" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-216853" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-247832" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-278811" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-309790" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-340769" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-371748" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-402727" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-433706" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-464685" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-495664" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-526643" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-557622" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-588601" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-619580" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"

#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-185874" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-216853" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-247832" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-278811" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-309790" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-340769" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-371748" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-402727" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-433706" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-464685" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-495664" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-526643" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-557622" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-588601" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"
#mount_model "philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-619580" "$EXP_DE_3_EXACT_MODELS_BASE_DIR/"

docker build \
  --build-arg TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_VERSION \
  --build-arg TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_LOG_LEVEL \
  --build-arg TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_VARIANT \
  -t ${DOCKER_REGISTRY}${TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_NAME}:${TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_VERSION}${TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_VARIANT}${TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_CUDA} \
  -f "src/main/docker/Dockerfile${TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_ANNOTATOR_CUDA}" \
  .

# unmount external models
#for model_mount_dir in "${model_mount_dirs[@]}"; do
#  echo "unmounting $model_mount_dir"
#  fusermount -u "$model_mount_dir"
#done
