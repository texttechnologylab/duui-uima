FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/resources/TypeSystemMM.xml ./TypeSystemMM.xml
COPY ./src/main/python/duui-mm.py ./duui-mm.py
COPY ./src/main/python/duui-mmduui-mm.lua ./duui-mm.lua

# dependencies
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# download models
RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForVision2Seq.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True)"


# log level
ARG DUUI_MM_LOG_LEVEL="DEBUG"
ENV DUUI_MM_LOG_LEVEL=$DUUI_MM_LOG_LEVEL

# config
ARG DUUI_MM_MODEL_CACHE_SIZE=3
ENV DUUI_MM_MODEL_CACHE_SIZE=$DUUI_MM_MODEL_CACHE_SIZE

# meta data
ARG DUUI_MM_ANNOTATOR_NAME="duui-mm"
ENV DUUI_MM_ANNOTATOR_NAME=$DUUI_MM_ANNOTATOR_NAME
ARG DUUI_MM_ANNOTATOR_VERSION="unset"
ENV DUUI_MM_ANNOTATOR_VERSION=$DUUI_MM_ANNOTATOR_VERSION

# Model Info
ARG DUUI_MM_MODEL_VERSION=0.1
ENV DUUI_MM_MODEL_VERSION=$DUUI_MM_MODEL_VERSION




ENTRYPOINT ["uvicorn", "duui-mm:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]