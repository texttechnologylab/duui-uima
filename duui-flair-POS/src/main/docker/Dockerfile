ARG CUDA_VERSION_PATCH="11.7.1"
ARG PYTHON_VERSION="3.10"
ARG PYTORCH_VERSION="1.13.1"

FROM docker.texttechnologylab.org/micromamba/flair:cuda-${CUDA_VERSION_PATCH}-python-${PYTHON_VERSION}-pytorch-${PYTORCH_VERSION}

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/bin/:$PATH

WORKDIR /usr/src/app

# Pre-load Flair taggers
ENV FLAIR_CACHE_ROOT="/usr/src/app/cache"
RUN python -c "from flair.models import SequenceTagger; SequenceTagger.load('de-pos')"

COPY ./src/main/lua/communication_layer.lua ./communication_layer.lua
COPY ./src/main/resources/dkpro-core-types.xml ./dkpro-core-types.xml
COPY ./src/main/resources/logging.yaml ./logging.yaml
COPY ./src/main/python/wsgi.py ./wsgi.py

ARG MODEL_CACHE_SIZE=1
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE
ARG FLAIR_BATCH_SIZE=128
ENV FLAIR_BATCH_SIZE=$FLAIR_BATCH_SIZE

ENTRYPOINT ["uvicorn", "wsgi:app", "--host", "0.0.0.0", "--port" ,"9714", "--log-config", "logging.yaml", "--use-colors"]
CMD ["--workers", "1"]
