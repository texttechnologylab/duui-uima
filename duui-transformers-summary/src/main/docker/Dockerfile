FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel


RUN pip install symspellpy fastapi uvicorn[standard] dkpro-cassis
RUN pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install torchserve==0.7.1
RUN pip install transformers==4.9.1
RUN pip install chardet==5.1.0
RUN pip install numpy
RUN pip install scikit-learn==0.22.1
RUN pip install pyhocon graphviz tensorboard pydantic
RUN pip install tqdm==4.56.0

# log level
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_LOG_LEVEL=$TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_LOG_LEVEL

# config
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_MODEL_CACHE_SIZE=1
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_MODEL_CACHE_SIZE=$TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_MODEL_CACHE_SIZE

# meta data
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_ANNOTATOR_NAME="textimager-duui-transformers-summary"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_ANNOTATOR_NAME=$TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_ANNOTATOR_NAME
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_ANNOTATOR_VERSION="0.0.1"
ENV TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_ANNOTATOR_VERSION=$TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_ANNOTATOR_VERSION

# copy scripts
COPY ./src/main/python/ ./
RUN ls -la ./*

ENTRYPOINT ["uvicorn", "textimager_duui_transformers_summary:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
