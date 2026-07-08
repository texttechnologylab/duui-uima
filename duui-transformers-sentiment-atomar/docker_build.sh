export ANNOTATOR_NAME=duui-transformers-sentiment-atomar
export ANNOTATOR_VERSION=0.5.1
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

###---------------------------------------------------------------------
#export MODEL_NAME="cardiffnlp/twitter-xlm-roberta-base-sentiment"
#export MODEL_SPECNAME="twitter-xlm-roberta-base-sentiment"
#export MODEL_VERSION="f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8"
#export MODEL_SOURCE="https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"
#export MODEL_SPECNAME="citizenlab-twitter-xlm-roberta-base-sentiment-finetunned"
#export MODEL_VERSION="a9381f1d9e6f8aac74155964c2f6ea9a63a9e9a6"
#export MODEL_SOURCE="https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
#export MODEL_SPECNAME="distilbert-student"
#export MODEL_VERSION="cf991100d706c13c0a080c097134c05b7f436c45"
#export MODEL_SOURCE="https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student"
#export MODEL_LANG="Multi"
###--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="philschmid/distilbert-base-multilingual-cased-sentiment"
#export MODEL_SPECNAME="distilbert-multilingual"
#export MODEL_VERSION="b45a713783e49ac09c94dfda4bff847f4ad771c5"
#export MODEL_SOURCE="https://huggingface.co/philschmid/distilbert-base-multilingual-cased-sentiment/tree/main"
#export MODEL_LANG="Multi"
####--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="cardiffnlp/twitter-roberta-base-sentiment-latest"
#export MODEL_SPECNAME="cardiffnlp-sentiment-en"
#export MODEL_VERSION="4ba3d4463bd152c9e4abd892b50844f30c646708"
#export MODEL_SOURCE="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="j-hartmann/sentiment-roberta-large-english-3-classes"
#export MODEL_SPECNAME="roberta-based-en"
#export MODEL_VERSION="81cdc0fe3eee1bc18d95ffdfb56b2151a39c9007"
#export MODEL_SOURCE="https://huggingface.co/j-hartmann/sentiment-roberta-large-english-3-classes"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="bardsai/finance-sentiment-de-base"
#export MODEL_SPECNAME="finance-sentiment-de"
#export MODEL_VERSION="51b3d03f716eaa093dc42130f675839675a07b9a"
#export MODEL_SOURCE="https://huggingface.co/bardsai/finance-sentiment-de-base"
#export MODEL_LANG="DE"
##--------------------------------------------------------------------

####---------------------------------------------------------------------
#export MODEL_NAME="oliverguhr/german-sentiment-bert"
#export MODEL_SPECNAME="german-sentiment-bert"
#export MODEL_VERSION="b1177ff59e305c966836ba2825d3dc2efc53f125"
#export MODEL_SOURCE="https://huggingface.co/oliverguhr/german-sentiment-bert"
#export MODEL_LANG="DE"
####--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="bardsai/finance-sentiment-zh-base"
#export MODEL_SPECNAME="finance-sentiment-zh"
#export MODEL_VERSION="33595d152578da080c6e5c94b60eba15a769107f"
#export MODEL_SOURCE="https://huggingface.co/bardsai/finance-sentiment-zh-base"
#export MODEL_LANG="ZH"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="bardsai/finance-sentiment-zh-fast"
#export MODEL_SPECNAME="finance-sentiment-zh-fast"
#export MODEL_VERSION="4cf6d7f85579bc73ac402d1dc4ecbcf3de8b6b7a"
#export MODEL_SOURCE="https://huggingface.co/bardsai/finance-sentiment-zh-fast"
#export MODEL_LANG="ZH"
##--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="bardsai/finance-sentiment-fr-base"
#export MODEL_SPECNAME="finance-sentiment-fr"
#export MODEL_VERSION="98f660ba2ca64140df78c1a29b91dc8b6beafb62"
#export MODEL_SOURCE="https://huggingface.co/bardsai/finance-sentiment-fr-base"
#export MODEL_LANG="FR"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="bardsai/twitter-sentiment-pl-base"
#export MODEL_SPECNAME="twitter-sentiment-pl-base"
#export MODEL_VERSION="612331865c33e03b87522600ca34b1425c400e90"
#export MODEL_SOURCE="https://huggingface.co/bardsai/twitter-sentiment-pl-base"
#export MODEL_LANG="PL"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="bardsai/twitter-sentiment-pl-fast"
#export MODEL_SPECNAME="twitter-sentiment-pl-fast"
#export MODEL_VERSION="2adf843ad928baf1d631179b4d52930fc286eee9"
#export MODEL_SOURCE="https://huggingface.co/bardsai/twitter-sentiment-pl-fast"
#export MODEL_LANG="PL"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="climatebert/distilroberta-base-climate-sentiment"
export MODEL_SPECNAME="distilroberta-base-climate-sentiment"
export MODEL_VERSION="e9f9a94ee4263f5ad5cfc97b8539a497fc88aa7d"
export MODEL_SOURCE="https://huggingface.co/climatebert/distilroberta-base-climate-sentiment"
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
