ARG GEONAMES_FST_VERSION=0.3.1

FROM docker.texttechnologylab.org/duui-geonames-fst/base:${GEONAMES_FST_VERSION} AS builder
WORKDIR /build/
RUN cargo build --release --no-default-features --features duui
RUN chmod +x /build/target/release/geonames-fst

FROM alpine:latest AS data
RUN apk --update add unzip && rm -rf /var/cache/apk/*
ARG COUNTRY="DE"
ADD https://download.geonames.org/export/dump/${COUNTRY}.zip /data/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/${COUNTRY}.zip /data/alternateNames/
WORKDIR /data/geonames
RUN unzip ${COUNTRY}.zip && rm -f ${COUNTRY}.zip readme.txt
WORKDIR /data/alternateNames
RUN unzip ${COUNTRY}.zip && rm -f ${COUNTRY}.zip readme.txt

FROM python:3.12-slim AS prod
COPY --from=builder /build/target/release/geonames-fst /app/
COPY --from=data /data /app/data/
COPY src/main/resources/ /app/resources/
WORKDIR /app/
RUN pip install --no-cache-dir -r /app/resources/requirements.txt \
    && chmod +x /app/resources/entrypoint.sh
ENV RUST_LOG="info,tower_http=debug,axum::rejection=trace"
ENV PORT="9714"
ENV GEONAMES_BACKEND_PORT="9715"
ENV GEONAMES_BACKEND="http://127.0.0.1:9715"
EXPOSE 9714
ENTRYPOINT ["/app/resources/entrypoint.sh"]
CMD []
