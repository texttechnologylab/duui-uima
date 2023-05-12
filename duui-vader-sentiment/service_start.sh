DUUI_VADER_SENTIMENT_ANNOTATOR_NAME="duui-vader-sentiment" \
DUUI_VADER_SENTIMENT_ANNOTATOR_VERSION="unset" \
DUUI_VADER_SENTIMENT_LOG_LEVEL="DEBUG" \
uvicorn src.main.python.textimager_duui_vader_sentiment:app --host 0.0.0.0 --port 9714 --workers 1
