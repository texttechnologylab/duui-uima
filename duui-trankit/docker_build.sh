export MODEL_NAME="xlm-roberta-base"
export CUDA=0


export TEXTIMAGER_TRANKIT_ANNOTATOR_NAME=duui-trankit
export TEXTIMAGER_TRANKIT_ANNOTATOR_VERSION=0.1

docker build \
  --build-arg MODEL_NAME \
  --build-arg CUDA \
  --build-arg TEXTIMAGER_TRANKIT_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_TRANKIT_ANNOTATOR_VERSION \
  -t ${TEXTIMAGER_TRANKIT_ANNOTATOR_NAME}:${TEXTIMAGER_TRANKIT_ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile \
  .