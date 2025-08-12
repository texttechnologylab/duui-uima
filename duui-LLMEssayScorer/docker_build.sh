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

###---------------------------------------------------------------------
#export MODEL_NAME="JacobLinCool/IELTS_essay_scoring_safetensors"
#export MODEL_SPECNAME="ielts-essay-scoring"
#export MODEL_VERSION="8d8d0193ec4cfcbc9c780fc6e71810dca7ee41f6"
#export MODEL_SOURCE="https://huggingface.co/JacobLinCool/IELTS_essay_scoring_safetensors"
#export MODEL_LANG="EN"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="BeGradingScorer"
#export MODEL_SPECNAME="be-grading-scorer"
#export MODEL_VERSION="0.0.1"
#export MODEL_SOURCE="https://link.springer.com/article/10.1007/s00521-024-10449-y"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSlowScorer"
#export MODEL_SPECNAME="llmaesslow-scorer"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-norubrics"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------


###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-zeroshot-1"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-zeroshot-2"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-zeroshot-3"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-zeroshot-4"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-zeroshot-5"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-zeroshot-6"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-zeroshot-7"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------
#
###---------------------------------------------------------------------
#export MODEL_NAME="LLMAESSScorer"
#export MODEL_SPEC_NAME="llmaes-scorer-zeroshot-8"
#export MODEL_VERSION="b0716f4ba9d9d6e3e6ed734f343aa4a5a24607e5"
#export MODEL_SOURCE="https://github.com/Xiaochr/LLM-AES"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="AAGScorer"
#export MODEL_SPEC_NAME="aag-scorer"
#export MODEL_VERSION="0.0.1"
#export MODEL_SOURCE="https://arxiv.org/pdf/2501.14305"
#export MODEL_LANG="-"
###---------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="GradingMedicalEducation"
export MODEL_SPEC_NAME="grading-medical-education"
export MODEL_VERSION="0.0.1"
export MODEL_SOURCE="https://bmcmededuc.biomedcentral.com/articles/10.1186/s12909-024-06026-5"
export MODEL_LANG="-"
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
  --build-arg MODEL_SPEC_NAME \
  --build-arg MODEL_LANG \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPEC_NAME}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  -f src/main/docker/Dockerfile${DUUI_CUDA} \
  .

docker tag \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPEC_NAME}:${ANNOTATOR_VERSION}${DUUI_CUDA} \
  ${DOCKER_REGISTRY}${ANNOTATOR_NAME}"-"${MODEL_SPEC_NAME}:latest${DUUI_CUDA}