export ANNOTATOR_NAME=duui-factchecking
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

##---------------------------------------------------------------------
export MODEL_NAME="coreferee"
export MODEL_SPECNAME="coreferee"
export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
export MODEL_LANG="EN,DE,FR,PL"
export MODEL_VARIANT="SM"
##--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="coreferee"
#export MODEL_SPECNAME="coreferee"
#export MODEL_VERSION="3ee6f2781e54988d6c3593c6b8f37cc3bae8f982"
#export MODEL_SOURCE="https://github.com/richardpaulhudson/coreferee"
#export MODEL_LANG="EN,DE,FR,PL"
#export MODEL_VARIANT="LG"
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
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}"-"${MODEL_VARIANT}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  -f src/main/docker/Dockerfile${DUUI_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}"-"${MODEL_VARIANT}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}"-"${MODEL_VARIANT}:latest${DUUI_CUDA}