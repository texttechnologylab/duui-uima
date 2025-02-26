FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/python/TypeSystemTextToTimage.xml ./TypeSystemTextToTimage.xml
COPY ./src/main/python/duui_text_to_image.py ./duui_text_to_image.py
COPY ./src/main/python/duui_text_to_image.lua ./duui_text_to_image.lua

# dependencies
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# download models
RUN python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('OFA-Sys/small-stable-diffusion-v0')"
#RUN python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('black-forest-labs/FLUX.1-dev')"


# log level
ARG TEXT_TO_IMAGE_LOG_LEVEL="DEBUG"
ENV TEXT_TO_IMAGE_LOG_LEVEL=$TEXT_TO_IMAGE_LOG_LEVEL

# config
ARG TEXT_TO_IMAGE_MODEL_CACHE_SIZE=3
ENV TEXT_TO_IMAGE_MODEL_CACHE_SIZE=$TEXT_TO_IMAGE_MODEL_CACHE_SIZE

# meta data
ARG TEXT_TO_IMAGE_ANNOTATOR_NAME="duui-text-to-image"
ENV TEXT_TO_IMAGE_ANNOTATOR_NAME=$TEXT_TO_IMAGE_ANNOTATOR_NAME
ARG TEXT_TO_IMAGE_ANNOTATOR_VERSION="unset"
ENV TEXT_TO_IMAGE_ANNOTATOR_VERSION=$TEXT_TO_IMAGE_ANNOTATOR_VERSION

# Model Info
ARG TEXT_TO_IMAGE_MODEL_VERSION=0.1
ENV TEXT_TO_IMAGE_MODEL_VERSION=$TEXT_TO_IMAGE_MODEL_VERSION


ARG TEXT_TO_IMAGE_HUGGING_FACE_TOKEN="unset"
ENV TEXT_TO_IMAGE_HUGGING_FACE_TOKEN=$TEXT_TO_IMAGE_HUGGING_FACE_TOKEN
# offline mode for huggingface
#ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE




ENTRYPOINT ["uvicorn", "duui_text_to_image:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
