# Stage 1: Builder
FROM python:3.10-slim AS builder
WORKDIR /usr/src/app

LABEL org.opencontainers.image.title="DUUI-Component for duprisal detection"
LABEL org.opencontainers.image.description="DUUI component for surprise detection in association with projects in the CRC NegLab (1629)"
LABEL org.opencontainers.image.authors="Giuseppe Abrami (abrami@em.uni-frankfurt.de), Mercedes Martinez Bruera (MartinezBruera@em.uni-frankfurt.de)"
LABEL org.opencontainers.image.source="https://github.com/texttechnologylab/duui-uima/tree/main/duui-suprisal"
LABEL org.opencontainers.image.licenses="AGPL-3.0"


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

#COPY requirements-base.txt ./
#RUN pip install --prefix=/install -r requirements-base.txt
#
#COPY requirements-ml.txt ./
#RUN pip install --prefix=/install -r requirements-ml.txt


# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /usr/src/app

LABEL org.opencontainers.image.title="DUUI-Component for duprisal detection"
LABEL org.opencontainers.image.description="DUUI component for surprise detection in association with projects in the CRC NegLab (1629)"
LABEL org.opencontainers.image.authors="Giuseppe Abrami (abrami@em.uni-frankfurt.de), Mercedes Martinez Bruera (MartinezBruera@em.uni-frankfurt.de)"
LABEL org.opencontainers.image.source="https://github.com/texttechnologylab/duui-uima/tree/main/duui-suprisal"
LABEL org.opencontainers.image.licenses="AGPL-3.0"


EXPOSE 9714

COPY --from=builder /install /usr/local

COPY ./src/main/docker/duui/communication.lua ./communication.lua
COPY ./src/main/docker/duui/typesystem.xml ./typesystem.xml
COPY ./src/main/docker/python/duui_suprisal.py ./duui_suprisal.py

ENTRYPOINT ["uvicorn", "duui_suprisal:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]
