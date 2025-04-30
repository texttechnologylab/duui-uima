ARG GEONAMES_FST_VERSION=0.3.1
FROM docker.texttechnologylab.org/duui-geonames-fst/base:${GEONAMES_FST_VERSION} AS builder

WORKDIR /build/
RUN cargo build --release --no-default-features --features duui
RUN chmod +x /build/target/release/geonames-fst

FROM alpine:latest AS data
RUN apk --update add unzip && rm -rf /var/cache/apk/*

ADD https://download.geonames.org/export/dump/AT.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/AT.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/BE.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/BE.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/BG.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/BG.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/CH.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/CH.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/CY.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/CY.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/CZ.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/CZ.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/DE.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/DE.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/DK.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/DK.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/EE.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/EE.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/GR.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/GR.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/ES.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/ES.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/FI.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/FI.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/FR.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/FR.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/HR.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/HR.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/HU.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/HU.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/IE.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/IE.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/IT.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/IT.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/LT.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/LT.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/LU.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/LU.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/LV.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/LV.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/MT.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/MT.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/NL.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/NL.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/NO.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/NO.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/PL.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/PL.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/PT.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/PT.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/RO.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/RO.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/SE.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/SE.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/SI.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/SI.zip /tmp/alternateNames/

ADD https://download.geonames.org/export/dump/SK.zip /tmp/geonames/
ADD https://download.geonames.org/export/dump/alternatenames/SK.zip /tmp/alternateNames/

RUN mkdir -p /data/geonames /data/alternateNames && \
    for COUNTRY in AT BE BG CH CY CZ DE DK EE ES FI FR GR HR HU IE IT LT LU LV MT NL NO PL PT RO SE SI SK; do \
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