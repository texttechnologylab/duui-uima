# All
export TEXTIMAGER_SPACY_VARIANT=
# Tokenization
#export TEXTIMAGER_SPACY_VARIANT=-tokenizer
# Sentence segmentation
#export TEXTIMAGER_SPACY_VARIANT=-sentencizer
# Lemmatization
#export TEXTIMAGER_SPACY_VARIANT=-lemmatizer
# Part of Speech (POS) tagging
#export TEXTIMAGER_SPACY_VARIANT=-tagger
# Named Entity Recognition (NER)
#export TEXTIMAGER_SPACY_VARIANT=-ner
# Dependency parsing (DEP)
#export TEXTIMAGER_SPACY_VARIANT=-parser
# Morphological features and coarse-grained POS
#export TEXTIMAGER_SPACY_VARIANT=-morphologizer

# extra name
export TEXTIMAGER_SPACY_NAME_EXTRA=-benepar-en-de-fr

export TEXTIMAGER_SPACY_ANNOTATOR_NAME=duui-spacy${TEXTIMAGER_SPACY_VARIANT}${TEXTIMAGER_SPACY_NAME_EXTRA}
export TEXTIMAGER_SPACY_ANNOTATOR_VERSION=0.5.1
export TEXTIMAGER_SPACY_LOG_LEVEL=DEBUG
export TEXTIMAGER_SPACY_MODEL_CACHE_SIZE=3

export DOCKER_REGISTRY="docker.texttechnologylab.org/"

docker build \
  --build-arg TEXTIMAGER_SPACY_VARIANT \
  --build-arg TEXTIMAGER_SPACY_ANNOTATOR_NAME \
  --build-arg TEXTIMAGER_SPACY_ANNOTATOR_VERSION \
  --build-arg TEXTIMAGER_SPACY_LOG_LEVEL \
  -t ${DOCKER_REGISTRY}${TEXTIMAGER_SPACY_ANNOTATOR_NAME}:${TEXTIMAGER_SPACY_ANNOTATOR_VERSION} \
  -f src/main/docker/Dockerfile \
  .

docker tag \
  ${DOCKER_REGISTRY}${TEXTIMAGER_SPACY_ANNOTATOR_NAME}:${TEXTIMAGER_SPACY_ANNOTATOR_VERSION} \
  ${DOCKER_REGISTRY}${TEXTIMAGER_SPACY_ANNOTATOR_NAME}:latest
