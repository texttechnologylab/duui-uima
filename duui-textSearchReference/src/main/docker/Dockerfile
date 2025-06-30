FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt


# service script
COPY ./src/main/python/TypeSystemTextSearchReference.xml ./TypeSystemTextSearchReference.xml
COPY ./src/main/python/extractReferenceText.py ./extractReferenceText.py
COPY ./src/main/python/duui_TextSearchReference.lua ./duui_TextSearchReference.lua
COPY ./src/main/python/duui_TextSearchReference.py ./duui_TextSearchReference.py


# meta data
ARG ANNOTATOR_NAME="duui-textsearchreference:app"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL


ENTRYPOINT ["uvicorn", "duui_TextSearchReference:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

