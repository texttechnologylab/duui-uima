FROM python:3.11-slim

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

# torch (with CUDA) in its own layer, the same in duui-wtpsplit, so that the images share it
RUN pip install --no-cache-dir torch==2.14.1

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# model at a fixed revision, downloaded at build time, the container then runs offline; chonky loads the tokenizer
# without a revision ("main"), so the build fails unless "main" is the pinned revision
ARG DUUI_CHONKY_MODEL_NAME="mirth/chonky_mmbert_small_multilingual_1"
ENV DUUI_CHONKY_MODEL_NAME=$DUUI_CHONKY_MODEL_NAME
ARG DUUI_CHONKY_MODEL_REVISION="4a92de5574053676cbc5fd06bac86a6e13588000"
ENV DUUI_CHONKY_MODEL_REVISION=$DUUI_CHONKY_MODEL_REVISION
RUN python -c "from huggingface_hub import snapshot_download; \
path = snapshot_download('$DUUI_CHONKY_MODEL_NAME', revision='main'); \
assert path.rstrip('/').endswith('$DUUI_CHONKY_MODEL_REVISION'), 'main is not the pinned revision: ' + path"
ENV HF_HUB_OFFLINE=1

ARG DUUI_CHONKY_LOG_LEVEL="INFO"
ENV DUUI_CHONKY_LOG_LEVEL=$DUUI_CHONKY_LOG_LEVEL

ARG DUUI_CHONKY_ANNOTATOR_NAME="duui-chonky"
ENV DUUI_CHONKY_ANNOTATOR_NAME=$DUUI_CHONKY_ANNOTATOR_NAME
ARG DUUI_CHONKY_ANNOTATOR_VERSION="unset"
ENV DUUI_CHONKY_ANNOTATOR_VERSION=$DUUI_CHONKY_ANNOTATOR_VERSION

COPY ./src/main/resources/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/python/duui.py ./duui.py
COPY ./src/main/lua/communication.lua ./communication.lua

ENTRYPOINT ["uvicorn", "duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
