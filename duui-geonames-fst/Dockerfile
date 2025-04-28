FROM rust:slim AS builder

WORKDIR /app/src/
ADD https://github.com/manu-schaaf/geonames-fst.git /app/src/
RUN cargo build --release --no-default-features --features duui
RUN chmod +x /app/src/target/release/geonames-fst

FROM alpine:latest AS data
RUN apk --update add unzip && rm -rf /var/cache/apk/*

ARG COUNTRY="DE"

ADD https://download.geonames.org/export/dump/${COUNTRY}.zip /data/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/${COUNTRY}.zip /data/alternateNames/

WORKDIR /data/geonames
RUN unzip ${COUNTRY}.zip && rm -f ${COUNTRY}.zip readme.txt

WORKDIR /data/alternateNames
RUN unzip ${COUNTRY}.zip && rm -f ${COUNTRY}.zip readme.txt

FROM cgr.dev/chainguard/glibc-dynamic:latest AS prod
COPY --from=builder /app/src/target/release/geonames-fst /app/
COPY --from=builder /app/src/resources /app/resources
COPY --from=data /data /app/data
WORKDIR /app/

ENV RUST_LOG="info,tower_http=debug,axum::rejection=trace"

EXPOSE 9714
ENTRYPOINT ["/app/geonames-fst", "--port", "9714", "/app/data/geonames/", "--alternate", "/app/data/alternateNames/"]
CMD []