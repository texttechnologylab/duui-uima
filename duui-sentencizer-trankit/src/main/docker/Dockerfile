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

# requirements
COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install -r requirements.txt

# load-model
RUN python -c "from trankit import Pipeline; p = Pipeline('auto')"
RUN python -c "from trankit import Pipeline; p = Pipeline('english')"
RUN python -c "from trankit import Pipeline; p = Pipeline('german')"

# ---------------------------------------------------------
# config
# ---------------------------------------------------------
# model name:
ARG DUUI_SENTENCIZER_TRANKIT_MODEL_NAME="xlm-roberta-base"
ENV DUUI_SENTENCIZER_TRANKIT_MODEL_NAME=$DUUI_SENTENCIZER_TRANKIT_MODEL_NAME
# use cuda
ARG DUUI_SENTENCIZER_TRANKIT_CUDA=0
ENV DUUI_SENTENCIZER_TRANKIT_CUDA=$DUUI_SENTENCIZER_TRANKIT_CUDA

# ---------------------------------------------------------
# ---------------------------------------------------------
# meta data
# ---------------------------------------------------------
# name
ARG DUUI_SENTENCIZER_TRANKIT_ANNOTATOR_NAME="duui-sentencizer-trankit"
ENV DUUI_SENTENCIZER_TRANKIT_ANNOTATOR_NAME=$DUUI_SENTENCIZER_TRANKIT_ANNOTATOR_NAME
# version
ARG DUUI_SENTENCIZER_TRANKIT_ANNOTATOR_VERSION="unset"
ENV DUUI_SENTENCIZER_TRANKIT_ANNOTATOR_VERSION=$DUUI_SENTENCIZER_TRANKIT_ANNOTATOR_VERSION

# service script
COPY ./src/main/resources/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/lua/communication.lua ./communication.lua
COPY ./src/main/python/duui_trankit.py ./duui_trankit.py

# server
ENTRYPOINT ["uvicorn", "duui_trankit:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]