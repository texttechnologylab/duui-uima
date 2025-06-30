export ANNOTATOR_NAME=duui-offensive
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

#---------------------------------------------------------------------
#export  MODEL_NAME="Hate-speech-CNERG/bert-base-uncased-hatexplain"
#export MODEL_SPECNAME="cnerg-hatexplain"
#export MODEL_VERSION="e487c81b768c7532bf474bd5e486dedea4cf3848"
#export MODEL_SOURCE="https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain"
#export MODEL_LANG="EN"
#--------------------------------------------------------------------

#---------------------------------------------------------------------
#export  MODEL_NAME="Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
#export MODEL_SPECNAME="cnerg-hatexplain-rationale"
#export MODEL_VERSION="7b1a724a178c639a4b3446c0ff8f13d19be4f471"
#export MODEL_SOURCE="https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
#export MODEL_LANG="EN"
#--------------------------------------------------------------------

#---------------------------------------------------------------------
#export  MODEL_NAME="worldbank/naija-xlm-twitter-base-hate"
#export MODEL_SPECNAME="naija-xlm-t-base-hate"
#export MODEL_VERSION="49fe8d380c290260b73e16ea005454ee28b27e5f"
#export MODEL_SOURCE="https://huggingface.co/worldbank/naija-xlm-twitter-base-hate"
#export MODEL_LANG="EN,HA,YO,IG,PIDGIN"
#--------------------------------------------------------------------

#---------------------------------------------------------------------
#export  MODEL_NAME="HateBERT_abuseval"
#export MODEL_SPECNAME="hatebert-abuseval"
#export MODEL_VERSION="d90e681c672a494bb555de99fc7ae780"
#export MODEL_SOURCE="https://osf.io/tbd58/files/osfstorage?view_only=d90e681c672a494bb555de99fc7ae780"
#export MODEL_LANG="EN"
#--------------------------------------------------------------------

#---------------------------------------------------------------------
#export  MODEL_NAME="pysentimiento/bertweet-hate-speech"
#export MODEL_SPECNAME="bertweet-hate-speech"
#export MODEL_VERSION="d9925de199f48face0d7026f07c3b492c423bbc0"
#export MODEL_SOURCE="https://huggingface.co/pysentimiento/bertweet-hate-speech"
#export MODEL_LANG="EN"
#--------------------------------------------------------------------

#---------------------------------------------------------------------
#export  MODEL_NAME="pysentimiento/robertuito-hate-speech"
#export MODEL_SPECNAME="robertuito-hate-speech"
#export MODEL_VERSION="db125ee7be2ad74457b900ae49a7e0f14f7a496c"
#export MODEL_SOURCE="https://huggingface.co/pysentimiento/robertuito-hate-speech"
#export MODEL_LANG="ES"
#--------------------------------------------------------------------

##---------------------------------------------------------------------
#export  MODEL_NAME="pysentimiento/bertabaporu-pt-hate-speech"
#export MODEL_SPECNAME="bertabaporu-hate-speech"
#export MODEL_VERSION="9d50687a13df38c7d2fdf4b2227eb28c006214de"
#export MODEL_SOURCE="https://huggingface.co/pysentimiento/bertabaporu-pt-hate-speech"
#export MODEL_LANG="PT"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export  MODEL_NAME="pysentimiento/bert-it-hate-speech"
#export MODEL_SPECNAME="bert-it-hate-speech"
#export MODEL_VERSION="627bbee98534e5bfbbc771fc6c7ecb35ffbfe90a"
#export MODEL_SOURCE="https://huggingface.co/pysentimiento/bert-it-hate-speech"
#export MODEL_LANG="IT"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export  MODEL_NAME="IMSyPP/hate_speech_multilingual"
#export MODEL_SPECNAME="imsypp-social-media"
#export MODEL_VERSION="2045782c975894635c4221a1d44aa23b24f0103e"
#export MODEL_SOURCE="https://huggingface.co/IMSyPP/hate_speech_multilingual"
#export MODEL_LANG="Multi"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export  MODEL_NAME="IMSyPP/hate_speech_en"
#export MODEL_SPECNAME="imsypp-social-media-en"
#export MODEL_VERSION="6dc7c7d81577a178a48d484f72cca334f44c7f69"
#export MODEL_SOURCE="https://huggingface.co/IMSyPP/hate_speech_en"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export  MODEL_NAME="IMSyPP/hate_speech_it"
#export MODEL_SPECNAME="imsypp-social-media-it"
#export MODEL_VERSION="46e36cd04dce8d3517b8014ce782ecc5306e2106"
#export MODEL_SOURCE="https://huggingface.co/IMSyPP/hate_speech_it"
#export MODEL_LANG="IT"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export  MODEL_NAME="IMSyPP/hate_speech_nl"
#export MODEL_SPECNAME="imsypp-social-media-nl"
#export MODEL_VERSION="571af0e4558288a3f1c249b5bfd1da8149a584a7"
#export MODEL_SOURCE="https://huggingface.co/IMSyPP/hate_speech_nl"
#export MODEL_LANG="NL"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="IMSyPP/hate_speech_slo"
#export MODEL_SPECNAME="imsypp-social-media-slo"
#export MODEL_VERSION="910059d15a0b554deb5591edc166015bd78848be"
#export MODEL_SOURCE="https://huggingface.co/IMSyPP/hate_speech_slo"
#export MODEL_LANG="SLO"
##--------------------------------------------------------------------

##---------------------------------------------------------------------
#export  MODEL_NAME="cardiffnlp/twitter-roberta-base-hate-multiclass-latest"
#export MODEL_SPECNAME="cardiffnlp-hate-multiclass"
#export MODEL_VERSION="b9a303f920f8527ac4151e65953c04505fdf0587"
#export MODEL_SOURCE="https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-multiclass-latest"
#export MODEL_LANG="EN"
##--------------------------------------------------------------------

#---------------------------------------------------------------------
export  MODEL_NAME="cardiffnlp/twitter-roberta-large-sensitive-multilabel"
export MODEL_SPECNAME="cardiffnlp-sensitive-multilabel"
export MODEL_VERSION="e362dc65d7042ac79d5893d250ba60be7d73ef39"
export MODEL_SOURCE="https://huggingface.co/cardiffnlp/twitter-roberta-large-sensitive-multilabel"
export MODEL_LANG="EN"
#--------------------------------------------------------------------

export DOCKER_REGISTRY="docker.texttechnologylab.org/"
export DUUI_CUDA=
#export DUUI_CUDA="-cuda"

docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  --build-arg MODEL_CACHE_SIZE \
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