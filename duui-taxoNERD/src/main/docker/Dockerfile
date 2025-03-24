#FROM python:3.10
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install setuptools wheel
RUN pip install setuptools pip
COPY ./requirements.txt ./requirements.txt
#RUN pip install --no-deps -r ./requirements.txt
RUN pip install -r ./requirements.txt

RUN pip install git+https://github.com/abrami/taxonerd
RUN pip install --no-binary :all: nmslib

# The possible models
RUN pip install https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_biobert-1.1.0.tar.gz
RUN pip install https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_biobert_weak-1.1.0.tar.gz
#RUN pip install https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_md-1.1.0.tar.gz
#RUN pip install https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_md_weak-1.1.0.tar.gz

# vars
#ARG MODEL="en_ner_eco_md"
ARG MODEL="en_ner_eco_biobert_weak"
#ARG MODEL="en_ner_eco_biobert"
ENV MODEL=$MODEL


# service script
COPY ./python/types.xml ./types.xml
COPY ./python/communication.lua ./communication.lua
COPY ./python/duui_taxonerd.py ./duui_taxonerd.py

ENTRYPOINT ["uvicorn", "duui_taxonerd:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]