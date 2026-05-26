ARG GEONAMES_FST_VERSION=0.3.1
FROM docker.texttechnologylab.org/duui-geonames-fst/base:${GEONAMES_FST_VERSION} AS builder

WORKDIR /build/
RUN cargo build --release --no-default-features --features duui
RUN chmod +x /build/target/release/geonames-fst

FROM alpine:latest AS data
RUN apk --update add unzip && rm -rf /var/cache/apk/*

ADD https://download.geonames.org/export/dump/AT.zip \
    https://download.geonames.org/export/dump/BE.zip \
    https://download.geonames.org/export/dump/BG.zip \
    https://download.geonames.org/export/dump/CH.zip \
    https://download.geonames.org/export/dump/CY.zip \
    https://download.geonames.org/export/dump/CZ.zip \
    https://download.geonames.org/export/dump/DE.zip \
    https://download.geonames.org/export/dump/DK.zip \
    https://download.geonames.org/export/dump/EE.zip \
    https://download.geonames.org/export/dump/GR.zip \
    https://download.geonames.org/export/dump/ES.zip \
    https://download.geonames.org/export/dump/FI.zip \
    https://download.geonames.org/export/dump/FR.zip \
    https://download.geonames.org/export/dump/HR.zip \
    https://download.geonames.org/export/dump/HU.zip \
    https://download.geonames.org/export/dump/IE.zip \
    https://download.geonames.org/export/dump/IT.zip \
    https://download.geonames.org/export/dump/LT.zip \
    https://download.geonames.org/export/dump/LU.zip \
    https://download.geonames.org/export/dump/LV.zip \
    https://download.geonames.org/export/dump/MT.zip \
    https://download.geonames.org/export/dump/NL.zip \
    https://download.geonames.org/export/dump/NO.zip \
    https://download.geonames.org/export/dump/PL.zip \
    https://download.geonames.org/export/dump/PT.zip \
    https://download.geonames.org/export/dump/RO.zip \
    https://download.geonames.org/export/dump/SE.zip \
    https://download.geonames.org/export/dump/SI.zip \
    https://download.geonames.org/export/dump/SK.zip \
    /tmp/geonames/

ADD https://download.geonames.org/export/dump/alternatenames/AT.zip \
    https://download.geonames.org/export/dump/alternatenames/BE.zip \
    https://download.geonames.org/export/dump/alternatenames/BG.zip \
    https://download.geonames.org/export/dump/alternatenames/CH.zip \
    https://download.geonames.org/export/dump/alternatenames/CY.zip \
    https://download.geonames.org/export/dump/alternatenames/CZ.zip \
    https://download.geonames.org/export/dump/alternatenames/DE.zip \
    https://download.geonames.org/export/dump/alternatenames/DK.zip \
    https://download.geonames.org/export/dump/alternatenames/EE.zip \
    https://download.geonames.org/export/dump/alternatenames/GR.zip \
    https://download.geonames.org/export/dump/alternatenames/ES.zip \
    https://download.geonames.org/export/dump/alternatenames/FI.zip \
    https://download.geonames.org/export/dump/alternatenames/FR.zip \
    https://download.geonames.org/export/dump/alternatenames/HR.zip \
    https://download.geonames.org/export/dump/alternatenames/HU.zip \
    https://download.geonames.org/export/dump/alternatenames/IE.zip \
    https://download.geonames.org/export/dump/alternatenames/IT.zip \
    https://download.geonames.org/export/dump/alternatenames/LT.zip \
    https://download.geonames.org/export/dump/alternatenames/LU.zip \
    https://download.geonames.org/export/dump/alternatenames/LV.zip \
    https://download.geonames.org/export/dump/alternatenames/MT.zip \
    https://download.geonames.org/export/dump/alternatenames/NL.zip \
    https://download.geonames.org/export/dump/alternatenames/NO.zip \
    https://download.geonames.org/export/dump/alternatenames/PL.zip \
    https://download.geonames.org/export/dump/alternatenames/PT.zip \
    https://download.geonames.org/export/dump/alternatenames/RO.zip \
    https://download.geonames.org/export/dump/alternatenames/SE.zip \
    https://download.geonames.org/export/dump/alternatenames/SI.zip \
    https://download.geonames.org/export/dump/alternatenames/SK.zip \
    /tmp/alternateNames/

RUN mkdir -p /data/geonames /data/alternateNames && \
    for COUNTRY in AT BE BG CH CY CZ DE DK EE ES FI FR GR HR HU IE IT LT LU LV MT NL NO PL PT RO SE SI SK; do \
        unzip -d /data/geonames/ /tmp/geonames/$COUNTRY.zip $COUNTRY.txt; \
        unzip -d /data/alternateNames/ /tmp/alternateNames/$COUNTRY.zip $COUNTRY.txt; \
    done

FROM cgr.dev/chainguard/glibc-dynamic:latest AS prod
COPY --from=builder /build/target/release/geonames-fst /app/
COPY --from=data /data /app/data/
COPY src/main/resources/ /app/resources/
WORKDIR /app/

ENV RUST_LOG="info,tower_http=debug,axum::rejection=trace"

EXPOSE 9714
ENTRYPOINT ["/app/geonames-fst", "--port", "9714", "/app/data/geonames/", "--alternate", "/app/data/alternateNames/"]
CMD []