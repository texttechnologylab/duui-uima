FROM python:3.11

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 9714


COPY ./src/main/docker/python/communication.lua ./communication.lua
COPY ./src/main/docker/python/duui_yt_dlp.py ./duui_yt_dlp.py
COPY ./src/main/docker/python/typesystem.xml ./typesystem.xml

ENTRYPOINT ["uvicorn", "duui_yt_dlp:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
