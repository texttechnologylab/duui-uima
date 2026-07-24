FROM debian:bookworm-slim

ARG MINIFORGE_VERSION=26.3.2-3
ARG TARGETARCH

# Installing Miniforge in a regular build layer avoids the extended filesystem
# attributes in the prebuilt Miniforge image, which some rootless Podman storage
# backends cannot unpack. Map Docker architecture names to Miniforge names.
RUN apt-get update && \
    apt-get install --yes --no-install-recommends ca-certificates curl bzip2 && \
    case "${TARGETARCH:-amd64}" in \
        amd64) miniforge_arch=x86_64 ;; \
        arm64) miniforge_arch=aarch64 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    curl --fail --location --retry 3 \
        --output /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${miniforge_arch}.sh" && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh && \
    /opt/conda/bin/conda clean --all --yes && \
    rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/opt/nltk_data \
    MPLCONFIGDIR=/tmp/matplotlib \
    CONDA_ENV_NAME=speaker-anonymization

ENV PATH=/opt/conda/bin:${PATH}

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
        portaudio19-dev \
        tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copy dependency metadata separately so application changes do not invalidate
# the expensive environment layer.
COPY environment.yml /tmp/environment.yml

RUN conda config --system --set channel_priority strict && \
    conda env create --file /tmp/environment.yml && \
    conda clean --all --yes && \
    rm -f /tmp/environment.yml

# Make the application environment the default without requiring `conda run`.
ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:${PATH}

# Catch the original ESPnet/typeguard failure while building the image rather
# than after the container has started.
RUN python -c 'from importlib.metadata import version; assert version("espnet") == "0.10.5"; assert version("typeguard") == "2.13.3"; from espnet2.bin.asr_inference import Speech2Text; print("ESPnet/typeguard compatibility check passed")' 

# Cache the NLTK data that the application otherwise downloads at startup.
RUN mkdir -p "${NLTK_DATA}" && \
    python -m nltk.downloader -d "${NLTK_DATA}" \
        averaged_perceptron_tagger \
        averaged_perceptron_tagger_eng \
        cmudict

ARG IMS_TOUCAN_RELEASE_URL=https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5

ENV MODELS_DIR=/usr/src/app/models \
    TORCH_HOME=/usr/src/app/models/torch

# Large release assets are checksum-pinned and fetched into the image rather
# than committed to Git.
COPY ./src/main/python/models/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py \
        --destination "${MODELS_DIR}" \
        --base-url "${IMS_TOUCAN_RELEASE_URL}" && \
    rm -f /tmp/download_models.py

ARG WHISPER_MODEL_ID=openai/whisper-large-v3
ARG WHISPER_MODEL_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1
ARG SILERO_VAD_REPOSITORY=snakers4/silero-vad:915dd3d639b8333a52e001af095f87c5b7f1e0ac

ENV WHISPER_MODEL_ID=${WHISPER_MODEL_ID} \
    WHISPER_MODEL_REVISION=${WHISPER_MODEL_REVISION} \
    SILERO_VAD_REPOSITORY=${SILERO_VAD_REPOSITORY}

# Cache exact Whisper and Silero revisions so runtime processing is offline and
# cannot silently change when upstream repositories move.
COPY ./src/main/docker/preload_external_models.py /tmp/preload_external_models.py
RUN python /tmp/preload_external_models.py \
        --models-dir "${MODELS_DIR}" \
        --whisper-model "${WHISPER_MODEL_ID}" \
        --whisper-revision "${WHISPER_MODEL_REVISION}" \
        --silero-repository "${SILERO_VAD_REPOSITORY}" && \
    rm -f /tmp/preload_external_models.py

ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/.cache/huggingface \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    NUMBA_CACHE_DIR=/tmp/.cache/numba

COPY ./src/main/python/ ./

RUN useradd --no-create-home --uid 10001 --shell /usr/sbin/nologin app

USER app

EXPOSE 9714

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:9714/v1/health", timeout=3).read()'

# Forward signals correctly to Uvicorn and reap orphaned child processes.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "duui_speaker_anonymization:app", "--host", "0.0.0.0", "--port", "9714"]
