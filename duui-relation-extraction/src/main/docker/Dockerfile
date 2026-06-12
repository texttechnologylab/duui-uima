FROM python:3.10

WORKDIR /app

COPY ../../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../python/TypSystemRelation.xml ./TypSystemRelation.xml
COPY ./src/main/python/duui_relation_extraction.py ./duui_relation_extraction.py
COPY ./src/main/python/duui_relation_extraction.lua ./duui_relation_extraction.lua
COPY ./src/main/python/relation_extraction.py ./relation_extraction.py

COPY ./src/main/python/offline_models/rebel_large ./offline_models/rebel_large
COPY ./src/main/python/offline_models/knowgl-large ./offline_models/knowgl-large

ARG MODEL_NAME="rebel_large"
ENV MODEL_NAME=$MODEL_NAME

EXPOSE 8000

ENV DEVICE=cpu

CMD ["uvicorn", "duui_relation_extraction:duui_relation_extraction", "--host", "0.0.0.0", "--port", "8000"]