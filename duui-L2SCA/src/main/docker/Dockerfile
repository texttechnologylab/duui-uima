FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt
RUN python -c "import stanza; stanza.download('en'); stanza.download('de');"

# service script
COPY ./src/main/python/TypeSystemL2SCA.xml ./TypeSystemL2SCA.xml
COPY ./src/main/python/L2SCA_Scorer.py ./L2SCA_Scorer.py
COPY ./src/main/python/duui_l2sca.lua ./duui_l2sca.lua
COPY ./src/main/python/duui_l2sca.py ./duui_l2sca.py


# meta data
ARG ANNOTATOR_NAME="duui-l2sca:app"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

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
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG=""
ENV MODEL_LANG=$MODEL_LANG



ENTRYPOINT ["uvicorn", "duui_l2sca:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

