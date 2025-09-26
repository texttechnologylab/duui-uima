FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./requirements.txt ./requirements.txt

RUN apt-get update

#RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./src/main/docker/duui/communication.lua ./communication.lua
COPY ./src/main/docker/duui/typesystem.xml ./typesystem.xml
COPY ./src/main/docker/python/duui_suprisal.py ./duui_suprisal.py


ENTRYPOINT ["uvicorn", "duui_suprisal:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
