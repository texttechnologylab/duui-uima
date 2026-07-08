FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu

COPY ./src/main/python/communication.lua ./communication.lua
COPY ./src/main/python/duui_anonymize.py ./duui_anonymize.py
COPY ./src/main/python/typesystem.xml ./typesystem.xml
COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "duui_anonymize:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]