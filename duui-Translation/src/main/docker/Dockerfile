FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714
RUN apt-get update && apt-get install -y protobuf-compiler
# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -c "from transformers import pipeline; pipeline('text-classification', model='facebook/mbart-large-50-many-to-many-mmt')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='facebook/nllb-200-distilled-600M')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='google/flan-t5-base')"
RUN python -c "import whisper; whisper.load_model('small')"

# copy scripts
COPY ./src/main/python/duui_translation.lua ./duui_translation.lua
COPY ./src/main/python/Translation.py ./Translation.py
COPY ./src/main/python/languages_flores200.json ./languages_flores200.json
COPY ./src/main/python/languages_MBArt.json ./languages_MBArt.json
COPY ./src/main/python/TypeSystemLanguage.xml ./TypeSystemLanguage.xml
COPY ./src/main/python/duui_translation.py ./duui_translation.py


# log level
ARG TRANSLATION_LOG_LEVEL="DEBUG"
ENV TRANSLATION_LOG_LEVEL=$TRANSLATION_LOG_LEVEL

# config
ARG TRANSLATION_MODEL_CACHE_SIZE=1
ENV TRANSLATION_MODEL_CACHE_SIZE=$TRANSLATION_MODEL_CACHE_SIZE

# meta data
ARG TRANSLATION_ANNOTATOR_NAME="duui-translation"
ENV TRANSLATION_ANNOTATOR_NAME=$TRANSLATION_ANNOTATOR_NAME
ARG TRANSLATION_ANNOTATOR_VERSION="unset"
ENV TRANSLATION_ANNOTATOR_VERSION=$TRANSLATION_ANNOTATOR_VERSION

ENTRYPOINT ["uvicorn", "duui_translation:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
