FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/python/config.yaml ./config.yaml
COPY ./src/main/python/TypeSystemBertTopic.xml ./TypeSystemUnifiedTopic.xml
COPY ./src/main/python/duui-transformers-berttopic.py ./duui-transformers-berttopic.py
COPY ./src/main/python/duui-transformers-berttopic.lua ./duui-transformers-berttopic.lua

# dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


ENTRYPOINT ["uvicorn", "duui-transformers-berttopic:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
