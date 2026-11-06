FROM python:3.11-slim

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

# torch (with CUDA) in its own layer, the same in duui-chonky, so that the images share it
RUN pip install --no-cache-dir torch==2.14.1

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# models at fixed revisions, downloaded at build time, the container then runs offline
ARG DUUI_WTPSPLIT_MODEL_NAME="sat-12l-sm"
ENV DUUI_WTPSPLIT_MODEL_NAME=$DUUI_WTPSPLIT_MODEL_NAME
ARG DUUI_WTPSPLIT_MODEL_REVISION="d70c72a9331b2d5a9e82baad00c64964a23a09bb"
ENV DUUI_WTPSPLIT_MODEL_REVISION=$DUUI_WTPSPLIT_MODEL_REVISION
ARG DUUI_WTPSPLIT_TOKENIZER_NAME="FacebookAI/xlm-roberta-base"
ENV DUUI_WTPSPLIT_TOKENIZER_NAME=$DUUI_WTPSPLIT_TOKENIZER_NAME
ARG DUUI_WTPSPLIT_TOKENIZER_REVISION="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
ENV DUUI_WTPSPLIT_TOKENIZER_REVISION=$DUUI_WTPSPLIT_TOKENIZER_REVISION
# SaT loads the model "segment-any-text/<model name>" from the Hub, the tokenizer needs only its config and vocabulary
# (same patterns as TOKENIZER_FILES in duui.py)
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('segment-any-text/$DUUI_WTPSPLIT_MODEL_NAME', revision='$DUUI_WTPSPLIT_MODEL_REVISION'); \
snapshot_download('$DUUI_WTPSPLIT_TOKENIZER_NAME', revision='$DUUI_WTPSPLIT_TOKENIZER_REVISION', allow_patterns=['*.json', '*.model'])"
ENV HF_HUB_OFFLINE=1

ARG DUUI_WTPSPLIT_LOG_LEVEL="INFO"
ENV DUUI_WTPSPLIT_LOG_LEVEL=$DUUI_WTPSPLIT_LOG_LEVEL

ARG DUUI_WTPSPLIT_ANNOTATOR_NAME="duui-wtpsplit"
ENV DUUI_WTPSPLIT_ANNOTATOR_NAME=$DUUI_WTPSPLIT_ANNOTATOR_NAME
ARG DUUI_WTPSPLIT_ANNOTATOR_VERSION="unset"
ENV DUUI_WTPSPLIT_ANNOTATOR_VERSION=$DUUI_WTPSPLIT_ANNOTATOR_VERSION

COPY ./src/main/resources/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/python/duui.py ./duui.py
COPY ./src/main/lua/communication.lua ./communication.lua

ENTRYPOINT ["uvicorn", "duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
