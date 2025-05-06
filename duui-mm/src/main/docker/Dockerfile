FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install other requirements
COPY falsh_attention_requirments.txt /falsh_attention_requirments.txt
RUN pip install --no-cache-dir -r /falsh_attention_requirments.txt

# Install flash-attention from source (with CUDA available)
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

RUN apt-get update && apt-get install -y libgl1-mesa-glx


EXPOSE 9714


# copy scripts
COPY ./src/main/resources/TypeSystemMM.xml ./TypeSystemMM.xml
COPY ./src/main/python/duui-mm.py ./duui-mm.py
COPY ./src/main/python/duui-mm.lua ./duui-mm.lua
COPY ./src/main/python/models/ ./models/

# dependencies
RUN pip install -U pip
RUN pip install setuptools wheel psutil packaging
RUN pip install --upgrade setuptools
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation

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

# CUDA
ENV CUDA_HOME=/usr/local/cuda




ENTRYPOINT ["uvicorn", "duui-mm:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]