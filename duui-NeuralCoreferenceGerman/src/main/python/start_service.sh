TEXTIMAGER_COREF_ANNOTATOR_NAME=srl_cuda \
TEXTIMAGER_COREF_ANNOTATOR_VERSION=0.0.1 \
TEXTIMAGER_COREF_LOG_LEVEL=DEBUG \
TEXTIMAGER_COREF_PARSER_MODEL_NAME=tueba10_electra_uncased \
uvicorn textimager_duui_coref_ger:app --host 0.0.0.0 --port 8501 --workers 1