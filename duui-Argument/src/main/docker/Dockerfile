FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714
RUN apt-get update && apt-get install -y protobuf-compiler
# dependencies
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -c "from transformers import pipeline; pipeline('text-classification', model='chkla/roberta-argument')"

# copy scripts
COPY ./src/main/python/duui_argument.lua ./duui_argument.lua
COPY ./src/main/python/ArgumentClassification.py ./ArgumentClassification.py
COPY ./src/main/python/ArgumentClassification.py ./ArgumentClassification.py
COPY ./src/main/python/TypeSystemArgument.xml ./TypeSystemArgument.xml
COPY ./src/main/python/ukp_classes.py ./ukp_classes.py
COPY /src/main/python/models ./models
COPY /src/main/python/duui_argument.py ./duui_argument.py

# log level
ARG ARGUMENT_LOG_LEVEL="DEBUG"
ENV ARGUMENT_LOG_LEVEL=$ARGUMENT_LOG_LEVEL

# config
ARG ARGUMENT_MODEL_CACHE_SIZE=1
ENV ARGUMENT_MODEL_CACHE_SIZE=$ARGUMENT_MODEL_CACHE_SIZE

# meta data
ARG ARGUMENT_ANNOTATOR_NAME="duui-argument"
ENV ARGUMENT_ANNOTATOR_NAME=$ARGUMENT_ANNOTATOR_NAME
ARG ARGUMENT_ANNOTATOR_VERSION="unset"
ENV ARGUMENT_ANNOTATOR_VERSION=$ARGUMENT_ANNOTATOR_VERSION
ARG CHATGPT_KEY=""
ENV CHATGPT_KEY=$CHATGPT_KEY

ENTRYPOINT ["uvicorn", "duui_argument:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
