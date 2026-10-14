FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg -y

RUN pip install --no-deps torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/main/docker/python/model_preloader.py ./model_preloader.py
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
RUN python -u ./model_preloader.py

COPY ./src/main/docker/python/typesystem.xml ./typesystem.xml
COPY ./src/main/docker/python/duui_whisperx.py ./duui_whisperx.py
COPY ./src/main/docker/python/communication.lua ./communication.lua

ARG DUUI_WHISPERX_LOG_LEVEL="DEBUG"
ENV DUUI_WHISPERX_LOG_LEVEL=$DUUI_WHISPERX_LOG_LEVEL

ARG DUUI_WHISPERX_ANNOTATOR_NAME="duui-whisperx"
ENV DUUI_WHISPERX_ANNOTATOR_NAME=$DUUI_WHISPERX_ANNOTATOR_NAME
ARG DUUI_WHISPERX_ANNOTATOR_VERSION="unset"
ENV DUUI_WHISPERX_ANNOTATOR_VERSION=$DUUI_WHISPERX_ANNOTATOR_VERSION

ENTRYPOINT ["uvicorn", "duui_whisperx:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
