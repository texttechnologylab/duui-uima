ARG PYTHON_IMAGE=python:3.12
FROM ${PYTHON_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /usr/src/app

EXPOSE 9714

# System dependencies. curl is needed only for model artifact downloads such as TEI2GO,
# but keeping it here makes the model-specific RUN block simple and deterministic.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies.
# IMPORTANT: docker_build.sh generates this requirements.txt per MODEL_NAME/MODEL_LANG,
# so each image installs only the dependencies needed for that one backend.
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# copy DUUI TimeX3 scripts
COPY ./src/main/python/TypeSystemTime.xml ./TypeSystemTime.xml
COPY ./src/main/python/duui_time.py ./duui_time.py
COPY ./src/main/python/time_recognition_backend.py ./time_recognition_backend.py
COPY ./src/main/python/duui_time.lua ./duui_time.lua

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config: one model per container
ARG MODEL_CACHE_SIZE=1
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE

# meta data
ARG ANNOTATOR_NAME="duui-time"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# Model info: exactly one model and one language per image.
ARG MODEL_NAME="microsoft"
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_SPECNAME="recognizers-text-suite"
ENV MODEL_SPECNAME=$MODEL_SPECNAME
ENV HF_TEMPORAL_MODEL_ID=$MODEL_SPECNAME
ARG MODEL_VERSION="0.1"
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG="de"
ENV MODEL_LANG=$MODEL_LANG

# Download/cache only the selected model artifact, never all model artifacts.
RUN set -eux; \
    model="$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')"; \
    lang="$(echo "${MODEL_LANG}" | tr '[:upper:]' '[:lower:]')"; \
    case "${model}" in \
      microsoft) \
        echo "Microsoft Recognizers-Text uses package data from requirements.txt; no extra model download."; \
        ;; \
      duckling|sutime) \
        echo "${MODEL_NAME} uses an external runtime service; no local model is downloaded."; \
        ;; \
      tei2go) \
        case "${lang}" in \
          de|en|es|fr|it|pt) tei2go_model="${lang}_tei2go" ;; \
          *) echo "Unsupported TEI2GO language: ${MODEL_LANG}" >&2; exit 1 ;; \
        esac; \
        curl -L \
          -o "/tmp/${tei2go_model}-0.0.0-py3-none-any.whl" \
          "https://huggingface.co/hugosousa/${tei2go_model}/resolve/main/${tei2go_model}-any-py3-none-any.whl"; \
        python -m pip install --no-deps "/tmp/${tei2go_model}-0.0.0-py3-none-any.whl"; \
        rm -f "/tmp/${tei2go_model}-0.0.0-py3-none-any.whl"; \
        ;; \
      timexy) \
        case "${lang}" in \
          de) spacy_model="de_core_news_sm" ;; \
          en) spacy_model="en_core_web_sm" ;; \
          fr) spacy_model="fr_core_news_sm" ;; \
          *) echo "Unsupported Timexy language: ${MODEL_LANG}. Supported: de en fr" >&2; exit 1 ;; \
        esac; \
        python -m spacy download "${spacy_model}"; \
        ;; \
      german-gelectra|bert-got-a-date|hf-token-classification) \
        hf_model="${MODEL_SPECNAME}"; \
        if [[ -z "${hf_model}" ]]; then \
          echo "MODEL_SPECNAME must contain a Hugging Face model id for ${MODEL_NAME}" >&2; \
          exit 1; \
        fi; \
        export HF_MODEL_ID="${hf_model}"; \
        python -c "import os; from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline; model_id=os.environ['HF_MODEL_ID']; print(f'Caching Hugging Face token-classification model: {model_id}'); AutoTokenizer.from_pretrained(model_id); AutoModelForTokenClassification.from_pretrained(model_id); pipeline('token-classification', model=model_id, aggregation_strategy='simple')"; \
        ;; \
      *) \
        echo "Unsupported MODEL_NAME=${MODEL_NAME}" >&2; \
        exit 1; \
        ;; \
    esac

# offline mode for HuggingFace/runtime after build-time model caching
ARG TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE
ENV HF_HUB_OFFLINE=$TRANSFORMERS_OFFLINE

ENTRYPOINT ["uvicorn", "duui_time:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]