export ANNOTATOR_NAME=duui-factchecking
export ANNOTATOR_VERSION=0.1.0
export LOG_LEVEL=INFO
eport MODEL_CACHE_SIZE=3

###---------------------------------------------------------------------
#export MODEL_NAME="unieval"
#export MODEL_SPECNAME="unieval"
#export MODEL_VERSION="d33e7b6cfebe97b2bafe435adbd818230d5a416a"
#export MODEL_SOURCE="https://github.com/maszhongming/UniEval"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="nubia"
#export MODEL_SPECNAME="nubia"
#export MODEL_VERSION="ba6569605671e88217a14b2b218ce6974be73775"
#export MODEL_SOURCE="https://github.com/wl-research/nubia"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

###---------------------------------------------------------------------
#export MODEL_NAME="manueldeprada/FactCC"
#export MODEL_SPECNAME="factcc"
#export MODEL_VERSION="c7b3148015d4ddc263f6e2acb2689e90ac061669"
#export MODEL_SOURCE="https://huggingface.co/manueldeprada/FactCC"
#export MODEL_LANG="EN"
###--------------------------------------------------------------------

##---------------------------------------------------------------------
export MODEL_NAME="MiniCheck"
export MODEL_SPECNAME="minicheck"
export MODEL_VERSION="75ea32eca40730ed76e161cdcd893ac87eddef48"
export MODEL_SOURCE="https://github.com/Liyan06/MiniCheck"
export MODEL_LANG="EN"
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