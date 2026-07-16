FROM quay.io/condaforge/miniforge3:26.3.2-3

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/opt/nltk_data \
    MPLCONFIGDIR=/tmp/matplotlib \
    CONDA_ENV_NAME=speaker-anonymization

# Runtime libraries used by the audio and speech packages, plus a compiler for
# pip packages that do not provide a wheel for this exact Python/platform pair.
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        espeak-ng \
        ffmpeg \
        libgl1 \
        libgomp1 \
        libportaudio2 \
        libsndfile1 \
        pkg-config \
        portaudio19-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY environment.yml /tmp/environment.yml

RUN conda config --system --set channel_priority strict && \
    conda env create --file /tmp/environment.yml && \
    conda clean --all --yes && \
    rm -f /tmp/environment.yml

ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:${PATH}

RUN python -c 'from importlib.metadata import version; assert version("espnet") == "0.10.5"; assert version("typeguard") == "2.13.3"; from espnet2.bin.asr_inference import Speech2Text; print("ESPnet/typeguard compatibility check passed")' 

RUN mkdir -p "${NLTK_DATA}" && \
    python -m nltk.downloader -d "${NLTK_DATA}" \
        averaged_perceptron_tagger \
        averaged_perceptron_tagger_eng \
        cmudict

COPY ./src/main/python/ ./

EXPOSE 9714


ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "duui_speaker_anonymization:app", "--host", "0.0.0.0", "--port", "9714"]
