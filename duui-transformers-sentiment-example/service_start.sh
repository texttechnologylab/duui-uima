TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_ANNOTATOR_NAME="duui-transformers-sentiment-example" \
TTLAB_DUUI_TRANSFORMERS_SENTIMENT_EXAMPLE_ANNOTATOR_VERSION="dev" \
uvicorn src.main.python.duui_transformers_sentiment:app --host 0.0.0.0 --port 9714 --workers 1
