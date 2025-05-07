FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


RUN apt update && \
    DEBIAN_FRONTEND=noninteractive \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 python3-pip python3-setuptools python3-distutils python3-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip


WORKDIR /usr/src/app

EXPOSE 9714 8000


RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg -y


# dependencies
RUN pip install -U pip
RUN pip install setuptools wheel psutil packaging torch==2.6.0 torchvision==0.21.0  torchaudio==2.6.0
RUN pip install --upgrade setuptools
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN echo "installing VLLM..."
RUN python -m pip install vllm

COPY ./src/main/python/setup_env.py ./setup_env.py
# Run setup_env.py and set environment variables
RUN python setup_env.py


# copy scripts
COPY ./src/main/resources/TypeSystemMM.xml ./TypeSystemMM.xml
COPY ./src/main/python/duui-mm.py ./duui-mm.py
COPY ./src/main/python/start.sh ./start.sh

COPY ./src/main/python/duui-mm.lua ./duui-mm.lua
COPY ./src/main/python/models/ ./models/



RUN chmod +x ./start.sh


RUN echo nvidia-smi

# log level
ARG MM_LOG_LEVEL="DEBUG"
ENV MM_LOG_LEVEL=$MM_LOG_LEVEL

# config
ARG MM_MODEL_CACHE_SIZE=3
ENV MM_MODEL_CACHE_SIZE=$MM_MODEL_CACHE_SIZE

# meta data
ARG MM_ANNOTATOR_NAME="duui-mutlimodality"
ENV MM_ANNOTATOR_NAME=$MM_ANNOTATOR_NAME
ARG MM_ANNOTATOR_VERSION="unset"
ENV MM_ANNOTATOR_VERSION=$MM_ANNOTATOR_VERSION

# Model Info
ARG MM_MODEL_VERSION=0.1
ENV MM_MODEL_VERSION=$MM_MODEL_VERSION

# VLLM logging level
ARG VLLM_LOGGING_LEVEL=DEBUG
ENV VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL


ENTRYPOINT ["./start.sh"]

#ENTRYPOINT ["/start.sh", "uvicorn", "duui-mm:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]