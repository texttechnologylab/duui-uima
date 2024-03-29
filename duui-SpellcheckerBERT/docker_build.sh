export TEXTIMAGER_SPBERT_ANNOTATOR_NAME=duui_spellcheck
export TEXTIMAGER_SPBERT_ANNOTATOR_VERSION=0.1.3
export TEXTIMAGER_SPBERT_LOG_LEVEL=INFO
eport TEXTIMAGER_SPBERT_MODEL_CACHE_SIZE=3
export  TEXTIMAGER_SPBERT_MODEL_NAME=spbert
export TEXTIMAGER_SPBERT_MODEL_VERSION=0.1
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

docker build \
  --build-arg TEXTIMAGER_SPBERT_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_SPBERT_ANNOTATOR_VERSION \
  --build-arg TEXTIMAGER_SPBERT_LOG_LEVEL \
  --build-arg TEXTIMAGER_SPBERT_MODEL_CACHE_SIZE \
  --build-arg TEXTIMAGER_SPBERT_MODEL_NAME \
  --build-arg TEXTIMAGER_SPBERT_MODEL_VERSION \
  -t ${DOCKER_REGISTRY}${TEXTIMAGER_SPBERT_ANNOTATOR_NAME}:${TEXTIMAGER_SPBERT_ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile \
  .

docker tag \
  ${DOCKER_REGISTRY}${TEXTIMAGER_SPBERT_ANNOTATOR_NAME}:${TEXTIMAGER_SPBERT_ANNOTATOR_VERSION} \
  ${DOCKER_REGISTRY}${TEXTIMAGER_SPBERT_ANNOTATOR_NAME}:latest