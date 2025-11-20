FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./src/main/python/LuminarAIDetection.xml ./LuminarAIDetection.xml
COPY ./src/main/python/main.py ./main.py
COPY ./src/main/python/luminar_ai_detection.lua ./luminar_ai_detection.lua

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Log level
ARG DUUI_LOG_LEVEL="DEBUG"
ENV DUUI_LOG_LEVEL=$DUUI_LOG_LEVEL

# Meta data
ARG DUUI_ANNOTATOR_NAME="duui-luminar-ai-detection"
ENV DUUI_ANNOTATOR_NAME=$DUUI_ANNOTATOR_NAME
ARG DUUI_ANNOTATOR_VERSION="0.1.0"
ENV DUUI_ANNOTATOR_VERSION=$DUUI_ANNOTATOR_VERSION

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
