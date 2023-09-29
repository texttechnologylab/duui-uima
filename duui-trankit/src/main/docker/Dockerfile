FROM nvidia/cuda:11.0.3-base-ubuntu20.04

RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install --no-install-recommends -y build-essential software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /usr/src/app

EXPOSE 9714

# ---------------------------------------------------------
# config
# ---------------------------------------------------------
# model name:
ARG MODEL_NAME="xlm-roberta-base"
ENV MODEL_NAME=$MODEL_NAME
# use cuda
ARG CUDA=0
ENV CUDA=$CUDA

# ---------------------------------------------------------
# ---------------------------------------------------------
# meta data
# ---------------------------------------------------------
# name
ARG TEXTIMAGER_TRANKIT_ANNOTATOR_NAME="duui-trankit"
ENV TEXTIMAGER_TRANKIT_ANNOTATOR_NAME=$TEXTIMAGER_TRANKIT_ANNOTATOR_NAME
# version
ARG TEXTIMAGER_TRANKIT_ANNOTATOR_VERSION=0.1
ENV TEXTIMAGER_TRANKIT_ANNOTATOR_VERSION=$TEXTIMAGER_TRANKIT_ANNOTATOR_VERSION

# service script
COPY ./src/main/python/communication.lua ./communication.lua
COPY ./src/main/python/dkpro-core-types.xml ./dkpro-core-types.xml
COPY ./src/main/python/duui_trankit.py ./duui_trankit.py
COPY ./requirements.txt ./requirements.txt

# requirements
RUN python3 -m pip install -r requirements.txt

# load-model
RUN python -c "from trankit import Pipeline; p = Pipeline('auto')"

# server
ENTRYPOINT ["uvicorn", "duui_trankit:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]