FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg -y

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

COPY ./src/main/python/crisper_whisper.py ./src/main/python/crisper_whisper.py
RUN python -u ./src/main/python/crisper_whisper.py

COPY ./src/main/resources/TypeSystem.xml ./src/main/resources/TypeSystem.xml
COPY ./src/main/python/duui.py ./src/main/python/duui.py
COPY ./src/main/lua/duui.lua ./src/main/lua/duui.lua

ARG DUUI_CRISPER_WHISPER_LOG_LEVEL="DEBUG"
ENV DUUI_CRISPER_WHISPER_LOG_LEVEL=$DUUI_CRISPER_WHISPER_LOG_LEVEL

ARG DUUI_CRISPER_WHISPER_ANNOTATOR_NAME="duui-whisperx"
ENV DUUI_CRISPER_WHISPER_ANNOTATOR_NAME=$DUUI_CRISPER_WHISPER_ANNOTATOR_NAME
ARG DUUI_CRISPER_WHISPER_ANNOTATOR_VERSION="unset"
ENV DUUI_CRISPER_WHISPER_ANNOTATOR_VERSION=$DUUI_CRISPER_WHISPER_ANNOTATOR_VERSION

ENTRYPOINT ["uvicorn", "src.main.python.duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]