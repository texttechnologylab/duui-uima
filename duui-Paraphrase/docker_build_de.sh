# OPTIONS: Lelon/t5-german-paraphraser-small or Lelon/t5-german-paraphraser-large
export MODEL_NAME=Lelon/t5-german-paraphraser-small
export CUDA=1
export GPU_ID=0


export TEXTIMAGER_PARA_ANNOTATOR_NAME=textimager-duui-paraphraser-de
export TEXTIMAGER_PARA_ANNOTATOR_VERSION=0.1

docker build \
  --build-arg MODEL_NAME \
  --build-arg CUDA \
  --build-arg GPU_ID \
  --build-arg TEXTIMAGER_PARA_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_PARA_ANNOTATOR_VERSION \
  -t ${TEXTIMAGER_PARA_ANNOTATOR_NAME}:${TEXTIMAGER_PARA_ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile_de \
  .

