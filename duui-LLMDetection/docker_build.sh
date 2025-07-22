export ANNOTATOR_NAME=duui-llmdetection
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

###---------------------------------------------------------------------
#export MODEL_NAME="Radar"
#export MODEL_SPECNAME="radar"
#export MODEL_VERSION="4ff1f23a69a36aa1df47b0933be6279f1b896c9b"
#export MODEL_SOURCE="https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B"
#export MODEL_LANG="Multi"
###---------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="HelloSimpleAI"
#export MODEL_SPECNAME="hellosimpleai"
#export MODEL_VERSION="d2b342c61775d5dd0221808a79983ed3b86ffd86"
#export MODEL_SOURCE="https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta"
#export MODEL_LANG="EN"
##---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="E5LoRA"
#export MODEL_SPECNAME="e5lora"
#export MODEL_VERSION="483fc4969592dc20e00e5130e7187b5dd25dbcc7"
#export MODEL_SOURCE="https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector"
#export MODEL_LANG="EN"
###---------------------------------------------------------------------

##---------------------------------------------------------------------
#export MODEL_NAME="Binocular"
#export MODEL_SPECNAME="binocular-falcon3-1b"
#export MODEL_VERSION="cb37ef3559b157b5c9d9226296ba01a5162da1f7,28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
#export MODEL_SOURCE="https://huggingface.co/tiiuae/Falcon3-1B-Base,https://huggingface.co/tiiuae/Falcon3-1B-Instrcut"
#export MODEL_LANG="Multi"
##---------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="DetectLLM-LRR"
#export MODEL_SPECNAME="detectllm-lrr-gpt2"
#export MODEL_VERSION="607a30d783dfa663caf39e06633721c8d4cfcd7e"
#export MODEL_SOURCE="https://huggingface.co/openai-community/gpt2"
#export MODEL_LANG="Multi"
###---------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="Fast-DetectGPT"
#export MODEL_SPECNAME="fast-detectgpt-gpt2"
#export MODEL_VERSION="607a30d783dfa663caf39e06633721c8d4cfcd7e"
#export MODEL_SOURCE="https://huggingface.co/openai-community/gpt2"
#export MODEL_LANG="Multi"
###---------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="Fast-DetectGPTwithScoring"
export MODEL_SPECNAME="fast-detectgpt-dif-falcon3-1"
export MODEL_VERSION="cb37ef3559b157b5c9d9226296ba01a5162da1f7,28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
export MODEL_SOURCE="https://huggingface.co/tiiuae/Falcon3-1B-Base,https://huggingface.co/tiiuae/Falcon3-1B-Instrcut"
export MODEL_LANG="Multi"
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
  --build-arg MODEL_LANG \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  -f src/main/docker/Dockerfile${DUUI_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPECNAME}:latest${DUUI_CUDA}