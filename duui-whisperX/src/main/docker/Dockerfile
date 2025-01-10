FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./src/main/docker/python/communication.lua ./communication.lua
COPY ./src/main/docker/python/duui_whisperx.py ./duui_whisperx.py
COPY ./src/main/docker/python/typesystem.xml ./typesystem.xml
COPY ./requirements.txt ./requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg -y

RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "duui_whisperx:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]