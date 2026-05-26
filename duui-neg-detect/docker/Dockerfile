FROM nvidia/cuda:12.8.1-base-ubuntu22.04

RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    curl \
    python3-pip \
    python3-setuptools \
    python3-distutils \
    git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Link python3 to python (optional, for convenience)
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip
RUN python3 -m pip install setuptools

WORKDIR /usr/src/app

EXPOSE 9714

# ---------------------------------------------------------
# config
# ---------------------------------------------------------
# nd-Version:
ARG NEG_DETECT_VERSION=0.1
ENV NEG_DETECT_VERSION=$NEG_DETECT_VERSION


# ---------------------------------------------------------
# ---------------------------------------------------------
# meta data
# ---------------------------------------------------------
# name
ARG TEXTIMAGER_ANNOTATOR_NAME="duui-neg-detect"
ENV TEXTIMAGER_ANNOTATOR_NAME=$TEXTIMAGER_ANNOTATOR_NAME
# version
ARG TEXTIMAGER_ANNOTATOR_VERSION=0.1
ENV TEXTIMAGER_ANNOTATOR_VERSION=$TEXTIMAGER_ANNOTATOR_VERSION
# cuda
ARG CUDA="cuda:0"
ENV CUDA=$CUDA
# cue_detection
ARG CUE_DETECTION="true"
ENV CUE_DETECTION=$CUE_DETECTION
# scope_detection
ARG SCOPE_DETECTION="true"
ENV SCOPE_DETECTION=$SCOPE_DETECTION
# lang
ARG LANG="de"
ENV LANG=$LANG

# service script
COPY ./requirements.txt ./requirements.txt


# requirements
RUN python3 -m pip install -r requirements.txt

COPY ./src/ ./src

WORKDIR /usr/src/app/src

# server
ENTRYPOINT ["uvicorn", "duui-neg-detect:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]