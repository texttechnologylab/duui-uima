
TEXTIMAGER_HANTA_ANNOTATOR_NAME=textimager_duui_hanta \
TEXTIMAGER_HANTA_ANNOTATOR_VERSION=0.0.1 \
TEXTIMAGER_HANTA_LOG_LEVEL=DEBUG \
TEXTIMAGER_HANTA_MODEL_NAME=morphmodel_ger.pgz \
uvicorn textimager_duui_hanta:app --host 0.0.0.0 --port 8501 --workers 1
