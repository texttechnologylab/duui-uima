FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714
RUN apt-get update && apt-get install -y protobuf-compiler
# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='kornosk/bert-election2020-twitter-stance-trump')"
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='mlburnham/deberta-v3-base-polistance-affect-v1.0')"

# copy scripts
COPY ./src/main/python/duui_stance.lua ./duui_stance.lua
COPY ./src/main/python/StanceDetection.py ./StanceDetection.py
COPY ./src/main/python/TypeSystemStance.xml ./TypeSystemStance.xml
COPY /src/main/python/duui_stance.py ./duui_stance.py

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config
ARG MODEL_CACHE_SIZE=1
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_VERSION=""
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_URL=""
ENV MODEL_URL=$MODEL_URL
ARG MODEL_LANG=""
ENV MODEL_LANG=$MODEL_LANG

# meta data
ARG ANNOTATOR_NAME="duui-stance"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION
ARG CHATGPT_KEY=""
ENV CHATGPT_KEY=$CHATGPT_KEY

ENTRYPOINT ["uvicorn", "duui_stance:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
