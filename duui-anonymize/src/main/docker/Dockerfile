FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu

COPY ./src/main/python/communication.lua ./communication.lua
COPY ./src/main/python/duui_anonymize.py ./duui_anonymize.py
COPY ./src/main/python/typesystem.xml ./typesystem.xml
COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "duui_anonymize:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]