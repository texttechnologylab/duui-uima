ARG GEONAMES_FST_VERSION=0.3.1
FROM docker.texttechnologylab.org/duui-geonames-fst/base:${GEONAMES_FST_VERSION} AS builder

WORKDIR /build/
RUN cargo build --release --no-default-features --features duui
RUN chmod +x /build/target/release/geonames-fst

FROM alpine:latest AS data
RUN apk --update add unzip && rm -rf /var/cache/apk/*

ADD https://download.geonames.org/export/dump/CZ.zip \
    https://download.geonames.org/export/dump/DK.zip \
    https://download.geonames.org/export/dump/DE.zip \
    https://download.geonames.org/export/dump/GB.zip \
    https://download.geonames.org/export/dump/HU.zip \
    https://download.geonames.org/export/dump/IE.zip \
    https://download.geonames.org/export/dump/PL.zip \
    https://download.geonames.org/export/dump/SK.zip \
    https://download.geonames.org/export/dump/SL.zip \
    /tmp/geonames/

ADD https://download.geonames.org/export/dump/alternatenames/CZ.zip \
    https://download.geonames.org/export/dump/alternatenames/DK.zip \
    https://download.geonames.org/export/dump/alternatenames/DE.zip \
    https://download.geonames.org/export/dump/alternatenames/GB.zip \
    https://download.geonames.org/export/dump/alternatenames/HU.zip \
    https://download.geonames.org/export/dump/alternatenames/IE.zip \
    https://download.geonames.org/export/dump/alternatenames/PL.zip \
    https://download.geonames.org/export/dump/alternatenames/SK.zip \
    https://download.geonames.org/export/dump/alternatenames/SL.zip \
    /tmp/alternateNames/

RUN mkdir -p /data/geonames /data/alternateNames && \
    for COUNTRY in CZ DK DE GB HU IE PL SK SL; do \
        unzip -d /data/geonames/ /tmp/geonames/$COUNTRY.zip $COUNTRY.txt; \
        unzip -d /data/alternateNames/ /tmp/alternateNames/$COUNTRY.zip $COUNTRY.txt; \
    done

FROM cgr.dev/chainguard/glibc-dynamic:latest AS prod
COPY --from=builder /build/target/release/geonames-fst /app/
COPY --from=data /data /app/data
COPY src/main/resources/ /app/resources
WORKDIR /app/

ENV RUST_LOG="info,tower_http=debug,axum::rejection=trace"

EXPOSE 9714
ENTRYPOINT ["/app/geonames-fst", "--port", "9714", "/app/data/geonames/", "--alternate", "/app/data/alternateNames/"]
CMD []