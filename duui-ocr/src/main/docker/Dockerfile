# Builds the container image for the multi-model OCR annotator.
# This is v3 of the Dockerfile.
#
#   v1:  Only downloaded PaddleOCR-VL-1.5. Straightforward, worked fine.
#
#   v2:  Added microsoft/trocr-base-printed to the pre-download step.
#        This actually *built* successfully, unlike the Python code which
#        never ran properly with TrOCR. I also had to add
#        VisionEncoderDecoderModel and TrOCRProcessor to the import
#        line, which was the moment I started realizing TrOCR was a
#        different "thing". The image was ~2GB larger for a model we never
#        ended up using. Removed it.
#
#   v3:  Replaced TrOCR with GLM-OCR. Downloads both PaddleOCR-VL and
#        GLM-OCR at build time. Current version.
#
# BORROWED. The overall structure (WORKDIR, EXPOSE, pip install pattern,
# ARG/ENV pairs for config) is copied from other DUUI annotator
# Dockerfiles in the TTLab repo:
# https://github.com/texttechnologylab/DockerUnifiedUIMAInterface
#
#
# Last meaningful edit: Feb 2026


# SOLID. Python 3.10 because that's what the TTLab DUUI components
# standardize on. Didn't investigate further, just went with 3.10.
FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

# -- Dependencies --
# SOLID. pip install in order: upgrade pip itself, install build tools
# (setuptools/wheel, needed for some packages that compile C extensions),
# then install from requirements.txt.

RUN pip install --upgrade pip
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# -- Model pre-download --
# SOLID (the idea) + COPILOT (the syntax)
#
# This downloads the model weights during the Docker *build*, not at
# runtime.
#
# v2 ABANDONED: this block used to also download TrOCR:
#
#   from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
#   VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed'); \
#   TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed'); \
#
# Removed it when I abandoned TrOCR. No point baking a ~900MB model
# into the image if we're never going to call it.
#
# Note: both working models (PaddleOCR-VL and GLM-OCR) load through
# the same Auto* classes AutoModelForImageTextToText and
# AutoProcessor. This is the same compatibility that makes them work
# in the shared Python backend. TrOCR needed VisionEncoderDecoderModel
# and TrOCRProcessor, which was another hint that it didn't belong here.
RUN python -c "\
from transformers import AutoProcessor, AutoModelForImageTextToText; \
AutoModelForImageTextToText.from_pretrained('PaddlePaddle/PaddleOCR-VL-1.5'); \
AutoProcessor.from_pretrained('PaddlePaddle/PaddleOCR-VL-1.5'); \
AutoModelForImageTextToText.from_pretrained('zai-org/GLM-OCR'); \
AutoProcessor.from_pretrained('zai-org/GLM-OCR')"

# -- Source files --
# SOLID. Copy the actual application code.
COPY ./src/main/python/TypeSystemOCR.xml ./TypeSystemOCR.xml
COPY ./src/main/python/duui_ocr.py ./duui_ocr.py
COPY ./src/main/python/duui_ocr.lua ./duui_ocr.lua

# -- Configuration --
# BORROWED. The ARG/ENV pattern is from TTLab's other Dockerfiles.

ARG DUUI_OCR_LOG_LEVEL="DEBUG"
ENV DUUI_OCR_LOG_LEVEL=$DUUI_OCR_LOG_LEVEL

# How many models to keep loaded in memory simultaneously.
# Default 1 means loading GLM-OCR evicts PaddleOCR and vice versa.
# Set to 2 if you have enough VRAM for both (~20GB+).
ARG DUUI_OCR_MODEL_CACHE_SIZE=1
ENV DUUI_OCR_MODEL_CACHE_SIZE=$DUUI_OCR_MODEL_CACHE_SIZE

# -- Metadata --
# These get reported through the /v1/documentation endpoint.
# The version is "unset" by default and gets overridden by the CI/CD
# pipeline (or manually with --build-arg).
ARG DUUI_OCR_ANNOTATOR_NAME="duui-ocr"
ENV DUUI_OCR_ANNOTATOR_NAME=$DUUI_OCR_ANNOTATOR_NAME
ARG DUUI_OCR_ANNOTATOR_VERSION="unset"
ENV DUUI_OCR_ANNOTATOR_VERSION=$DUUI_OCR_ANNOTATOR_VERSION

# -- Startup --
# BORROWED. uvicorn is the ASGI server that runs the FastAPI app.
#
# REVISIT. I've read that uvicorn with --workers > 1 uses separate
# processes, each with its own memory space. So two workers = two
# copies of the model in VRAM? That's definitely not what we want.
# But I wonder if there's a way to share the model across workers.
# Haven't looked into it. For now, 1 worker is fine for our throughput.
ENTRYPOINT ["uvicorn", "duui_ocr:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]