FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/python/config.yaml ./config.yaml
COPY ./src/main/python/TypeSystemUnifiedTopic.xml ./TypeSystemUnifiedTopic.xml
COPY ./src/main/python/duui-tts-converter.py ./duui-tts-converter.py
COPY ./src/main/python/duui-tts-converter.lua ./duui-tts-converter.lua

# dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


ENTRYPOINT ["uvicorn", "duui-tts-converter:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
