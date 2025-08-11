export ANNOTATOR_NAME=duui-essayscorer
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

###---------------------------------------------------------------------
#export MODEL_NAME="KevSun/Engessay_grading_ML"
#export MODEL_SPECNAME="engessay-grading-ml"
#export MODEL_VERSION="3de133614f5b91172eb35d3e1e93a8de4a38bad4"
#export MODEL_SOURCE="https://huggingface.co/KevSun/Engessay_grading_ML"
#export MODEL_LANG="EN"
###---------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="JacobLinCool/IELTS_essay_scoring_safetensors"
export MODEL_SPECNAME="ielts-essay-scoring"
export MODEL_VERSION="8d8d0193ec4cfcbc9c780fc6e71810dca7ee41f6"
export MODEL_SOURCE="https://huggingface.co/JacobLinCool/IELTS_essay_scoring_safetensors"
export MODEL_LANG="EN"
##---------------------------------------------------------------------

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
  --build-arg MODEL_SPECNAME \
  --build-arg MODEL_LANG \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  -f src/main/docker/Dockerfile${DUUI_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:latest${DUUI_CUDA}