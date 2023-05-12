FROM python:3.8

WORKDIR /usr/src/app

COPY ./src/main/docker/python/gervader ./gervader

RUN pip install --upgrade pip
RUN pip install setuptools wheel
COPY ./src/main/docker/requirements.txt ./requirements.txt
RUN pip install --no-deps -r ./requirements.txt

# service script
COPY ./src/main/docker/python/dkpro-core-types.xml ./dkpro-core-types.xml
COPY ./src/main/docker/python/communication.lua ./communication.lua
COPY ./src/main/docker/python/duui_gervader.py ./duui_gervader.py

ENTRYPOINT ["uvicorn", "duui_gervader:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "2"]