FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


RUN apt update && \
    DEBIAN_FRONTEND=noninteractive \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 python3-pip python3-setuptools python3-distutils python3-dev curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip


WORKDIR /usr/src/app

EXPOSE 9714 8000


RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg -y


# dependencies
RUN pip install setuptools wheel psutil packaging torch torchvision torchvision
RUN pip install --upgrade setuptools
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN python -m pip install vllm
RUN pip uninstall -y libtpu-nightly jax jaxlib





# copy scripts
COPY ./src/main/resources/TypeSystemVisionLanguage.xml ./TypeSystemVisionLanguage.xml
COPY ./src/main/python/duui-vision-language.py ./duui-vision-language.py
COPY ./src/main/python/start_vllm_server.sh ./start_vllm_server.sh

COPY ./src/main/python/duui-vision-language.lua ./duui-vision-language.lua
COPY ./src/main/python/models/ ./models/

RUN chmod +x ./start_vllm_server.sh

# log level
ARG MM_LOG_LEVEL="DEBUG"
ENV MM_LOG_LEVEL=$MM_LOG_LEVEL

# config
ARG MM_MODEL_CACHE_SIZE=3
ENV MM_MODEL_CACHE_SIZE=$MM_MODEL_CACHE_SIZE

# meta data
ARG MM_ANNOTATOR_NAME="duui-vision-language"
ENV MM_ANNOTATOR_NAME=$MM_ANNOTATOR_NAME
ARG MM_ANNOTATOR_VERSION="unset"
ENV MM_ANNOTATOR_VERSION=$MM_ANNOTATOR_VERSION

# Model Info
ARG MM_MODEL_VERSION=0.1
ENV MM_MODEL_VERSION=$MM_MODEL_VERSION

ARG VLLM_LOGGING_LEVEL=DEBUG
ENV VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL

ENV VLLM_PLATFORM=cuda


ENTRYPOINT ["./start_vllm_server.sh"]

#ENTRYPOINT ["/start.sh", "uvicorn", "duui-mm:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]