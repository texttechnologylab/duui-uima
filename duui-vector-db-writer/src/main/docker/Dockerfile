FROM python:3.13

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ARG DUUI_VECTOR_DB_WRITER_LOG_LEVEL="DEBUG"
ENV DUUI_VECTOR_DB_WRITER_LOG_LEVEL=$DUUI_VECTOR_DB_WRITER_LOG_LEVEL

ARG DUUI_VECTOR_DB_WRITER_ANNOTATOR_NAME="duui-vector-db-writer"
ENV DUUI_VECTOR_DB_WRITER_ANNOTATOR_NAME=$DUUI_VECTOR_DB_WRITER_ANNOTATOR_NAME
ARG DUUI_VECTOR_DB_WRITER_ANNOTATOR_VERSION="unset"
ENV DUUI_VECTOR_DB_WRITER_ANNOTATOR_VERSION=$DUUI_VECTOR_DB_WRITER_ANNOTATOR_VERSION

# Verbindungsdaten haben bewusst keine Build-ARG-Defaults fuer Passwort/Host --
# die werden beim "docker run" per -e gesetzt, nicht ins Image gebacken.
ENV DUUI_VECTOR_DB_WRITER_DB_PORT=5432

COPY ./src/main/resources/TypeSystem.xml ./src/main/resources/TypeSystem.xml
COPY ./src/main/python/duui.py ./src/main/python/duui.py
COPY ./src/main/lua/communication.lua ./src/main/lua/communication.lua

ENTRYPOINT ["uvicorn", "src.main.python.duui:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
