FROM nvidia/cuda:12.8.1-cudnn-runtime-ubi8 AS base

RUN dnf install -y python3.12 python3.12-pip && python3.12 -m pip install -U pip setuptools wheel

WORKDIR /app/
RUN python3.12 -m venv venv && ./venv/bin/python -m pip install -U 'spacy[cuda12x]' 'fastapi[standard]' uvicorn pydantic_settings spacy-lookups-data

RUN ./venv/bin/python -m spacy download ca_core_news_sm && \
    ./venv/bin/python -m spacy download zh_core_web_sm && \
    ./venv/bin/python -m spacy download hr_core_news_sm && \
    ./venv/bin/python -m spacy download da_core_news_sm && \
    ./venv/bin/python -m spacy download nl_core_news_sm && \
    ./venv/bin/python -m spacy download en_core_web_sm && \
    ./venv/bin/python -m spacy download fi_core_news_sm && \
    ./venv/bin/python -m spacy download fr_core_news_sm && \
    ./venv/bin/python -m spacy download de_core_news_sm && \
    ./venv/bin/python -m spacy download el_core_news_sm && \
    ./venv/bin/python -m spacy download it_core_news_sm && \
    ./venv/bin/python -m spacy download ja_core_news_sm && \
    ./venv/bin/python -m spacy download ko_core_news_sm && \
    ./venv/bin/python -m spacy download lt_core_news_sm && \
    ./venv/bin/python -m spacy download mk_core_news_sm && \
    ./venv/bin/python -m spacy download nb_core_news_sm && \
    ./venv/bin/python -m spacy download pl_core_news_sm && \
    ./venv/bin/python -m spacy download pt_core_news_sm && \
    ./venv/bin/python -m spacy download ro_core_news_sm && \
    ./venv/bin/python -m spacy download ru_core_news_sm && \
    ./venv/bin/python -m spacy download sl_core_news_sm && \
    ./venv/bin/python -m spacy download es_core_news_sm && \
    ./venv/bin/python -m spacy download sv_core_news_sm && \
    ./venv/bin/python -m spacy download uk_core_news_sm && \
    ./venv/bin/python -m spacy download xx_sent_ud_sm

COPY src/ /app/

ARG COMPONENT_NAME="duui-spacy-eos"
ENV COMPONENT_NAME=${COMPONENT_NAME}

ARG VERSION="0.1.0"
ENV COMPONENT_VERSION=${VERSION}

ARG SPACY_LANGUAGE="xx"
ENV SPACY_LANGUAGE=${SPACY_LANGUAGE}

ENTRYPOINT ["./venv/bin/uvicorn", "python.duui:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]