FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714
RUN apt-get update && apt-get install -y protobuf-compiler
# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -c "from transformers import pipeline; pipeline('text-classification', model='qanastek/51-languages-classifier')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')"
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='cis-lmu/glotlid', filename='model.bin')"
RUN python -m spacy download en_core_web_sm

# copy scripts
COPY ./src/main/python/duui_language.lua ./duui_language.lua
COPY ./src/main/python/LanguageDetection.py ./LanguageDetection.py
COPY ./src/main/python/lid.176.bin ./lid.176.bin
COPY ./src/main/python/TypeSystemLanguage.xml ./TypeSystemLanguage.xml
COPY ./src/main/python/duui_language_annotation.py ./duui_language_annotation.py


# log level
ARG LANGUAGE_LOG_LEVEL="DEBUG"
ENV LANGUAGE_LOG_LEVEL=$LANGUAGE_LOG_LEVEL

# config
ARG LANGUAGE_MODEL_CACHE_SIZE=1
ENV LANGUAGE_MODEL_CACHE_SIZE=$LANGUAGE_MODEL_CACHE_SIZE

# meta data
ARG LANGUAGE_ANNOTATOR_NAME="textimager-duui-transformers-topic"
ENV LANGUAGE_ANNOTATOR_NAME=$LANGUAGE_ANNOTATOR_NAME
ARG LANGUAGE_ANNOTATOR_VERSION="unset"
ENV LANGUAGE_ANNOTATOR_VERSION=$LANGUAGE_ANNOTATOR_VERSION


# offline mode for huggingface
#ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE

ENTRYPOINT ["uvicorn", "duui_language_annotation:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
