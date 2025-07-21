FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/resources/TypeSystemImageToText.xml ./TypeSystemImageToText.xml
COPY ./src/main/python/duui_image_to_text.py ./duui_image_to_text.py
COPY ./src/main/python/duui_image_to_text.lua ./duui_image_to_text.lua

# dependencies
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# download models
RUN python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('microsoft/kosmos-2-patch14-224')"
RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True)"


# log level
ARG IMAGE_TO_TEXT_LOG_LEVEL="DEBUG"
ENV IMAGE_TO_TEXT_LOG_LEVEL=$IMAGE_TO_TEXT_LOG_LEVEL

# config
ARG IMAGE_TO_TEXT_MODEL_CACHE_SIZE=3
ENV IMAGE_TO_TEXT_MODEL_CACHE_SIZE=$IMAGE_TO_TEXT_MODEL_CACHE_SIZE

# meta data
ARG IMAGE_TO_TEXT_ANNOTATOR_NAME="duui-image-to-text"
ENV IMAGE_TO_TEXT_ANNOTATOR_NAME=$IMAGE_TO_TEXT_ANNOTATOR_NAME
ARG IMAGE_TO_TEXT_ANNOTATOR_VERSION="unset"
ENV IMAGE_TO_TEXT_ANNOTATOR_VERSION=$IMAGE_TO_TEXT_ANNOTATOR_VERSION

# Model Info
ARG IMAGE_TO_TEXT_MODEL_VERSION=0.1
ENV IMAGE_TO_TEXT_MODEL_VERSION=$IMAGE_TO_TEXT_MODEL_VERSION


# offline mode for huggingface
#ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE




ENTRYPOINT ["uvicorn", "duui_image_to_text:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
