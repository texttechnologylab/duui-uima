FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/app
EXPOSE 9714

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    ffmpeg gcc g++ git \
    libsndfile1 libsox-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN pip install --upgrade pip

RUN pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install 'transformers>=4.30,<4.40' typeguard==2.13.3

COPY . /usr/src/speaker-anonymization

COPY ./src/main/python/duui_speaker_anonymization.py /usr/src/app/duui_speaker_anonymization.py
COPY ./src/main/python/communication.lua /usr/src/app/communication.lua
COPY ./src/main/python/typesystem.xml /usr/src/app/typesystem.xml

ENV ANON_REPO_ROOT=/usr/src/speaker-anonymization
ENV MODELS_DIR=/usr/src/speaker-anonymization/models

RUN pip install --no-deps -r /usr/src/speaker-anonymization/anonymization/modules/tts/IMSToucan/requirements.txt 2>/dev/null || true

ENTRYPOINT ["uvicorn", "duui_speaker_anonymization:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
