export FACT_ANNOTATOR_NAME=duui-FactChecking
export FACT_ANNOTATOR_VERSION=0.1.0
export FACT_LOG_LEVEL=INFO
eport FACT_MODEL_CACHE_SIZE=3
export  FACT_MODEL_NAME=spbert
export FACT_MODEL_VERSION=0.1
export DOCKER_REGISTRY="docker.texttechnologylab.org/"

docker build \
  --build-arg FACT_ANNOTATOR_NAME \
  --build-arg FACT_ANNOTATOR_VERSION \
  --build-arg FACT_LOG_LEVEL \
  --build-arg FACT_MODEL_CACHE_SIZE \
  --build-arg FACT_MODEL_NAME \
  --build-arg FACT_MODEL_VERSION \
  -t ${DOCKER_REGISTRY}${FACT_ANNOTATOR_NAME}:${FACT_ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile \
  .

docker tag \
  ${DOCKER_REGISTRY}${FACT_ANNOTATOR_NAME}:${FACT_ANNOTATOR_VERSION} \
  ${DOCKER_REGISTRY}${FACT_ANNOTATOR_NAME}:latest