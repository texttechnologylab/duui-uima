FROM python:3.11

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/python/config.yaml ./config.yaml
COPY ./src/main/python/TypeSystemBertTopic.xml ./TypeSystemUnifiedTopic.xml
COPY ./src/main/python/duui-transformers-berttopic.py ./duui-transformers-berttopic.py
COPY ./src/main/python/duui-transformers-berttopic.lua ./duui-transformers-berttopic.lua

# dependencies
COPY ./requirements.txt ./requirements.txt
RUN SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt


ENTRYPOINT ["uvicorn", "duui-transformers-berttopic:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
