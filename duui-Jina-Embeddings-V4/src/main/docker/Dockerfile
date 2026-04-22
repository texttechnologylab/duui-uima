FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./src/main/docker/python/communication.lua ./communication.lua
COPY ./src/main/docker/python/duui_jina.py ./duui_jina.py
COPY ./src/main/docker/python/typesystem.xml ./typesystem.xml
COPY ./requirements.txt ./requirements.txt

RUN apt-get update

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "duui_jina:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]