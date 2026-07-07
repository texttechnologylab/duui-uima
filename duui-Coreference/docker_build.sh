export ANNOTATOR_NAME=duui-coreference
export ANNOTATOR_VERSION=0.2.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

###---------------------------------------------------------------------
#export MODEL_NAME="coreferee"
#export MODEL_SPECNAME="coreferee"
#export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
#export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
#export MODEL_LANG="en"
#export MODEL_VARIANT="sm"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="coreferee"
export MODEL_SPECNAME="coreferee"
export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
export MODEL_LANG="de"
export MODEL_VARIANT="sm"
##--------------------------------------------------------------------
#
###---------------------------------------------------------------------
#export MODEL_NAME="coreferee"
#export MODEL_SPECNAME="coreferee"
#export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
#export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
#export MODEL_LANG="fr"
#export MODEL_VARIANT="sm"
###--------------------------------------------------------------------
#
###---------------------------------------------------------------------
#export MODEL_NAME="coreferee"
#export MODEL_SPECNAME="coreferee"
#export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
#export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
#export MODEL_LANG="pl"
#export MODEL_VARIANT="sm"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="coreferee"
#export MODEL_SPECNAME="coreferee"
#export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
#export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
#export MODEL_LANG="en"
#export MODEL_VARIANT="lg"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="coreferee"
#export MODEL_SPECNAME="coreferee"
#export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
#export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
#export MODEL_LANG="de"
#export MODEL_VARIANT="lg"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="coreferee"
#export MODEL_SPECNAME="coreferee"
#export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
#export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
#export MODEL_LANG="fr"
#export MODEL_VARIANT="lg"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="coreferee"
#export MODEL_SPECNAME="coreferee"
#export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
#export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
#export MODEL_LANG="pl"
#export MODEL_VARIANT="lg"
###--------------------------------------------------------------------


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
  --build-arg MODEL_VARIANT \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}"-"${MODEL_LANG}"-"${MODEL_VARIANT}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  -f src/main/docker/Dockerfile${DUUI_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}"-"${MODEL_LANG}"-"${MODEL_VARIANT}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}"-"${MODEL_LANG}"-"${MODEL_VARIANT}:latest${DUUI_CUDA}