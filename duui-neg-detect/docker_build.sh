export NEG_DETECT_VERSION=3.3

export TEXTIMAGER_ANNOTATOR_NAME=duui-neg-detect
export TEXTIMAGER_ANNOTATOR_VERSION=0.1

export CUDA="cuda:0"
export CUE_DETECTION=true
export SCOPE_CUE_DETECTION=true
export LANG="de"

docker build \
  --build-arg NEG_DETECT_VERSION \
  --build-arg TEXTIMAGER_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_ANNOTATOR_VERSION \
  --build-arg CUDA \
  --build-arg CUE_DETECTION \
  --build-arg SCOPE_CUE_DETECTION \
  --build-arg LANG \
  -t ${TEXTIMAGER_ANNOTATOR_NAME}:${TEXTIMAGER_ANNOTATOR_VERSION} \
  -f docker/Dockerfile \
  .
