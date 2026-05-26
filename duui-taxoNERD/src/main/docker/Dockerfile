# Build Stage
FROM python:3.10-slim AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev cmake git \
 && rm -rf /var/lib/apt/lists/*
COPY ./src/main/docker/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt \
 && pip install --user --no-cache-dir git+https://github.com/abrami/taxonerd \
 && pip install --user --no-cache-dir --no-binary :all: nmslib

RUN pip install --user https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_md-1.1.0.tar.gz \
    && pip install --user https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_md_weak-1.1.0.tar.gz \
    && pip install --user https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_biobert-1.1.0.tar.gz \
    && pip install --user https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_biobert_weak-1.1.0.tar.gz



# Runtime Stage
FROM python:3.10-slim
WORKDIR /usr/src/app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY ./src/main/docker/python/ ./

ENTRYPOINT ["uvicorn", "duui_taxonerd:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
