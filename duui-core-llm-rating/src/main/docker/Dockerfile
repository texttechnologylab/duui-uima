FROM python:3.11

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

ARG DUUI_CORE_LLM_RATING_LOG_LEVEL="DEBUG"
ENV DUUI_CORE_LLM_RATING_LOG_LEVEL=$DUUI_CORE_LLM_RATING_LOG_LEVEL

ARG DUUI_CORE_LLM_RATING_ANNOTATOR_NAME="duui-core-llm-rating"
ENV DUUI_CORE_LLM_RATING_ANNOTATOR_NAME=$DUUI_CORE_LLM_RATING_ANNOTATOR_NAME
ARG DUUI_CORE_LLM_RATING_ANNOTATOR_VERSION="unset"
ENV DUUI_CORE_LLM_RATING_ANNOTATOR_VERSION=$DUUI_CORE_LLM_RATING_ANNOTATOR_VERSION

COPY ./src/main/resources/TypeSystem.xml ./src/main/resources/TypeSystem.xml
COPY ./src/main/python/duui.py ./src/main/python/duui.py
COPY ./src/main/lua/communication.lua ./src/main/lua/communication.lua

ENTRYPOINT ["uvicorn", "src.main.python.duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
