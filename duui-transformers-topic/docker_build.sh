export ANNOTATOR_NAME=duui-transformers-topic
export ANNOTATOR_VERSION=0.4.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3
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

##---------------------------------------------------------------------
export MODEL_NAME="WebOrganizer/TopicClassifier"
export MODEL_SPECNAME="organize-web"
export MODEL_VERSION="8d158c9d514cdc21a7c8e9bd94e5dc483d49e024"
export MODEL_SOURCE="https://huggingface.co/WebOrganizer/TopicClassifier"
export MODEL_LANG="EN"
##--------------------------------------------------------------------

export DOCKER_REGISTRY="docker.texttechnologylab.org/"
export DUUI_CUDA=

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
