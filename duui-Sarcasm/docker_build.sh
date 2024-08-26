export ANNOTATOR_NAME=duui-sarcasm
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

##---------------------------------------------------------------------
export MODEL_NAME="helinivan/multilingual-sarcasm-detector"
export MODEL_SPECNAME="multilingual-sarcasm-detector"
export MODEL_VERSION="832f28ed6d8bb7bca494ef1f504aa71fc077cf13"
export MODEL_SOURCE="https://huggingface.co/helinivan/multilingual-sarcasm-detector"
export MODEL_LANG="Multi"
##--------------------------------------------------------------------

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