FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

ARG DUUI_SENTENCIZER_CORENLP_CNLPVERSION="4.5.5"
ENV DUUI_SENTENCIZER_CORENLP_CNLPVERSION=$DUUI_SENTENCIZER_CORENLP_CNLPVERSION
RUN wget "https://nlp.stanford.edu/software/stanford-corenlp-${DUUI_SENTENCIZER_CORENLP_CNLPVERSION}.zip" && unzip "stanford-corenlp-${DUUI_SENTENCIZER_CORENLP_CNLPVERSION}.zip" && rm "stanford-corenlp-${DUUI_SENTENCIZER_CORENLP_CNLPVERSION}.zip"
ENV CORENLP_HOME="./stanford-corenlp-${DUUI_SENTENCIZER_CORENLP_CNLPVERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends openjdk-11-jre && apt-get clean && rm -rf /var/tmp/* /tmp/* /var/lib/apt/lists/*

RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

ARG DUUI_SENTENCIZER_CORENLP_LOG_LEVEL="DEBUG"
ENV DUUI_SENTENCIZER_CORENLP_LOG_LEVEL=$DUUI_SENTENCIZER_CORENLP_LOG_LEVEL

ARG DUUI_SENTENCIZER_CORENLP_ANNOTATOR_NAME="duui-sentencizer-corenlp"
ENV DUUI_SENTENCIZER_CORENLP_ANNOTATOR_NAME=$DUUI_SENTENCIZER_CORENLP_ANNOTATOR_NAME
ARG DUUI_SENTENCIZER_CORENLP_ANNOTATOR_VERSION="unset"
ENV DUUI_SENTENCIZER_CORENLP_ANNOTATOR_VERSION=$DUUI_SENTENCIZER_CORENLP_ANNOTATOR_VERSION

COPY ./src/main/resources/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/python/duui.py ./duui.py
COPY ./src/main/lua/communication.lua ./communication.lua

ENTRYPOINT ["uvicorn", "duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
