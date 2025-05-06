FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714 8000


# copy scripts
COPY ./src/main/resources/TypeSystemMM.xml ./TypeSystemMM.xml
COPY ./src/main/python/duui-mm.py ./duui-mm.py
COPY ./src/main/python/start.sh ./start.sh

COPY ./src/main/python/duui-mm.lua ./duui-mm.lua
COPY ./src/main/python/models/ ./models/

RUN chmod +x ./start.sh



# dependencies
RUN pip install -U pip
RUN pip install setuptools wheel psutil packaging
RUN pip install --upgrade setuptools
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN #pip install flash-attn --no-build-isolation

# download models (not working for attention models)
RUN #python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True, revision='0af439b3adb8c23fda473c4f86001dbf9a226021', device_map='cuda', torch_dtype='auto', _attn_implementation='eager')"


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

ENV CUDA_HOME=/usr/local/cuda



# CUDA
ENV CUDA_HOME=/usr/local/cuda




ENTRYPOINT ["./start.sh"]

#ENTRYPOINT ["/start.sh", "uvicorn", "duui-mm:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]