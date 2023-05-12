TEXTIMAGER_SPACY_ANNOTATOR_NAME="textimager_duui_spellcheck:app" \
TEXTIMAGER_SPACY_ANNOTATOR_VERSION="unset" \
TEXTIMAGER_SPACY_LOG_LEVEL="DEBUG" \
TEXTIMAGER_SPACY_MODEL_CACHE_SIZE="3" \
uvicorn textimager_duui_spacy:app --host 0.0.0.0 --port 9714
