FROM nvidia/cuda:12.8.1-cudnn-runtime-ubi8 AS base

RUN dnf install -y python3.12 python3.12-pip && python3.12 -m pip install -U pip setuptools wheel

WORKDIR /app/
RUN python3.12 -m venv venv && ./venv/bin/python -m pip install -U 'spacy[cuda12x]' 'fastapi[standard]' uvicorn pydantic_settings spacy-lookups-data 'transformers[hf_xet]'

RUN ./venv/bin/python -m spacy download de_core_news_sm && \
    ./venv/bin/python -m spacy download de_core_news_md && \
    ./venv/bin/python -m spacy download de_core_news_lg && \
    ./venv/bin/python -m spacy download en_core_web_sm && \
    ./venv/bin/python -m spacy download en_core_web_md && \
    ./venv/bin/python -m spacy download en_core_web_lg && \
    ./venv/bin/python -m spacy download de_dep_news_trf

COPY src/main/ /app/

ARG COMPONENT_NAME="duui-spacy-lua-process"
ENV COMPONENT_NAME=${COMPONENT_NAME}

ARG VERSION="0.1.0"
ENV COMPONENT_VERSION=${VERSION}

ARG SPACY_LANGUAGE="de"
ENV SPACY_LANGUAGE=${SPACY_LANGUAGE}

ARG SPACY_BATCH_SIZE=32
ENV SPACY_BATCH_SIZE=${SPACY_BATCH_SIZE}

ARG REQUEST_BATCH_SIZE=1024
ENV REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE}

ENTRYPOINT ["./venv/bin/uvicorn", "wsgi:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]