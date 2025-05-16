FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sox \
    libsndfile1 \
    ffmpeg

ARG DUUI_CANARY_GIT_BRANCH="r2.3.0"
ENV DUUI_CANARY_GIT_BRANCH=$DUUI_CANARY_GIT_BRANCH

RUN pip install "git+https://github.com/NVIDIA/NeMo.git@$DUUI_CANARY_GIT_BRANCH#egg=nemo_toolkit[asr]"

ADD "https://raw.githubusercontent.com/NVIDIA/NeMo/refs/heads/$DUUI_CANARY_GIT_BRANCH/examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py" ./speech_to_text_aed_chunked_infer.py

RUN python -c "from nemo.collections.asr.models import EncDecMultiTaskModel; EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-flash')"

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./src/main/resources/typesystem.xml ./src/main/resources/typesystem.xml
COPY ./src/main/python/duui_canary.py ./src/main/python/duui_canary.py
COPY ./src/main/lua/communication.lua ./src/main/lua/communication.lua

ARG DUUI_CANARY_ANNOTATOR_NAME="duui-canary"
ENV DUUI_CANARY_ANNOTATOR_NAME=$DUUI_CANARY_ANNOTATOR_NAME
ARG DUUI_CANARY_ANNOTATOR_VERSION="unset"
ENV DUUI_CANARY_ANNOTATOR_VERSION=$DUUI_CANARY_ANNOTATOR_VERSION

ARG DUUI_CANARY_LOG_LEVEL="DEBUG"
ENV DUUI_CANARY_LOG_LEVEL=$DUUI_CANARY_LOG_LEVEL

ENTRYPOINT ["python", "-m", "uvicorn", "src.main.python.duui_canary:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]
