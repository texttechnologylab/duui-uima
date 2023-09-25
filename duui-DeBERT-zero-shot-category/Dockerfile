FROM python:3.8

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY src/main/python/requirements.txt src/main/python/requirements.txt
RUN pip install -r ./src/main/python/requirements.txt

COPY src/main/python/download_debert_zero_shot.py src/main/python/download_debert_zero_shot.py
RUN python3 src/main/python/download_debert_zero_shot.py

COPY src/main/python/dkpro-core-types.xml ./dkpro-core-types.xml
COPY src/main/python/communication.lua ./communication.lua
COPY src/main/python/debert_zero_shot.py ./debert_zero_shot.py

EXPOSE 9714

ENTRYPOINT ["uvicorn", "debert_zero_shot:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
