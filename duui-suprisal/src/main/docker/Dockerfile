# Stage 1: Builder
FROM python:3.10-slim AS builder
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt requirements-ml.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements-base.txt \
#    && pip install --prefix=/install --cache-dir=/pip-cache --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt
    && pip install -r requirements-ml.txt

# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /usr/src/app
EXPOSE 9714
COPY --from=builder /install /usr/local

COPY ./src/main/docker/duui/communication.lua ./communication.lua
COPY ./src/main/docker/duui/typesystem.xml ./typesystem.xml
COPY ./src/main/docker/python/duui_suprisal.py ./duui_suprisal.py

ENTRYPOINT ["uvicorn", "duui_suprisal:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
