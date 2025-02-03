FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

RUN DEBIAN_FRONTEND=noninteractive apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y python3-dev poppler-utils tesseract-ocr tesseract-ocr-deu zlib1g-dev

# downgrade pip due to error on textract install:
# ... invalid metadata: .* suffix can only be used with `==` or `!=` operators ...
RUN python -m pip install "pip<24.1"
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

ARG DUUI_PDF_EXTRACT_OCR_LOG_LEVEL="DEBUG"
ENV DUUI_PDF_EXTRACT_OCR_LOG_LEVEL=$DUUI_PDF_EXTRACT_OCR_LOG_LEVEL

ARG DUUI_PDF_EXTRACT_OCR_ANNOTATOR_NAME="duui-pdf-extract-ocr"
ENV DUUI_PDF_EXTRACT_OCR_ANNOTATOR_NAME=$DUUI_PDF_EXTRACT_OCR_ANNOTATOR_NAME
ARG DUUI_PDF_EXTRACT_OCR_ANNOTATOR_VERSION="unset"
ENV DUUI_PDF_EXTRACT_OCR_ANNOTATOR_VERSION=$DUUI_PDF_EXTRACT_OCR_ANNOTATOR_VERSION

COPY ./src/main/resources/TypeSystem.xml ./src/main/resources/TypeSystem.xml
COPY ./src/main/python/duui.py ./src/main/python/duui.py
COPY ./src/main/lua/communication.lua ./src/main/lua/communication.lua

ENTRYPOINT ["uvicorn", "src.main.python.duui:app", "--host", "0.0.0.0", "--port" ,"9714", "--use-colors"]
CMD ["--workers", "1"]
