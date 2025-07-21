FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


RUN apt update && \
    DEBIAN_FRONTEND=noninteractive \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip


WORKDIR /usr/src/app

EXPOSE 9714 8000


RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg -y


# dependencies
RUN pip install setuptools wheel psutil packaging
RUN pip install --upgrade setuptools

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# download models (not working for attention models)
RUN #python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True, revision='0af439b3adb8c23fda473c4f86001dbf9a226021', device_map='cuda', torch_dtype='auto', _attn_implementation='eager')"



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

ENV CUDA_HOME=/usr/local/cuda



# CUDA
ENV CUDA_HOME=/usr/local/cuda




ENTRYPOINT ["./start_vllm_server.sh"]

#ENTRYPOINT ["/start.sh", "uvicorn", "duui-mm:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]