FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install --upgrade pip
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


# copy scripts
COPY ./src/main/resources/typesystem.xml ./typesystem.xml
COPY ./src/main/python/duui-parliament-segmenter.py ./duui-parliament-segmenter.py
COPY ./src/main/python/duui-parliament-segmenter.lua ./duui-parliament-segmenter.lua
COPY ./src/main/python/mp.py ./mp.py

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# meta data
ARG ANNOTATOR_NAME="duui-parliament-segmenter"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION


ENTRYPOINT ["uvicorn", "duui-parliament-segmenter:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
