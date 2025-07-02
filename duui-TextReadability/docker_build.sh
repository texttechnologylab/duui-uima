export ANNOTATOR_NAME=duui-textreadability
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

#---------------------------------------------------------------------
#export  MODEL_NAME="Textstat"
#export MODEL_SPECNAME="textstat"
#export MODEL_VERSION="d47dc863cdff8db317b0999c8879a3d2ddf00f35"
#export MODEL_SOURCE="https://github.com/textstat/textstat"
#export MODEL_LANG="EN"
#--------------------------------------------------------------------

#---------------------------------------------------------------------
export MODEL_NAME="Diversity"
export MODEL_SPECNAME="diversity"
export MODEL_VERSION="0.2.2"
export MODEL_SOURCE="https://pypi.org/project/diversity/"
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
