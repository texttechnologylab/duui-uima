FROM python:3.8

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY src/main/python/requirements.txt src/main/python/requirements.txt
RUN pip install -r ./src/main/python/requirements.txt

COPY src/main/python/download_sentiment_bert_model.py src/main/python/download_sentiment_bert_model.py
RUN python3 src/main/python/download_sentiment_bert_model.py

COPY src/main/python/dkpro-core-types.xml ./dkpro-core-types.xml
COPY src/main/python/communication.lua ./communication.lua
COPY src/main/python/debert-zero-short.py ./duui_german_sentiment_bert.py

ENTRYPOINT ["uvicorn", "duui_german_sentiment_bert:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
