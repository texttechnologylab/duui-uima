# OPTIONS: humarin/chatgpt_paraphraser_on_T5_base , tuner007/pegasus_paraphrase , eugenesiow/bart-paraphrase or
#          prithivida/parrot_paraphraser_on_T5
export MODEL_NAME=humarin/chatgpt_paraphraser_on_T5_base
export CUDA=1
export GPU_ID=0


export TEXTIMAGER_PARA_ANNOTATOR_NAME=textimager-duui-paraphraser-en
export TEXTIMAGER_PARA_ANNOTATOR_VERSION=0.1

docker build \
  --build-arg MODEL_NAME \
  --build-arg CUDA \
  --build-arg GPU_ID \
  --build-arg TEXTIMAGER_PARA_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_PARA_ANNOTATOR_VERSION \
  -t ${TEXTIMAGER_PARA_ANNOTATOR_NAME}:${TEXTIMAGER_PARA_ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile_en \
  .

