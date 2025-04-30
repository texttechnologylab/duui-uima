ARG GEONAMES_FST_VERSION=0.3.1
FROM docker.texttechnologylab.org/duui-geonames-fst/base:${GEONAMES_FST_VERSION} AS builder

WORKDIR /build/
RUN cargo build --release --no-default-features --features duui
RUN chmod +x /build/target/release/geonames-fst

FROM alpine:latest AS data
RUN apk --update add unzip && rm -rf /var/cache/apk/*

ADD https://download.geonames.org/export/dump/DE.zip \
    https://download.geonames.org/export/dump/FR.zip \
    https://download.geonames.org/export/dump/GR.zip \
    https://download.geonames.org/export/dump/HU.zip \
    https://download.geonames.org/export/dump/BG.zip \
    https://download.geonames.org/export/dump/DK.zip \
    https://download.geonames.org/export/dump/EE.zip \
    https://download.geonames.org/export/dump/CH.zip \
    https://download.geonames.org/export/dump/CY.zip \
    https://download.geonames.org/export/dump/CZ.zip \
    https://download.geonames.org/export/dump/FI.zip \
    https://download.geonames.org/export/dump/ES.zip \
    https://download.geonames.org/export/dump/GB.zip \
    https://download.geonames.org/export/dump/IT.zip \
    https://download.geonames.org/export/dump/LI.zip \
    https://download.geonames.org/export/dump/LT.zip \
    https://download.geonames.org/export/dump/LU.zip \
    https://download.geonames.org/export/dump/LV.zip \
    https://download.geonames.org/export/dump/ME.zip \
    https://download.geonames.org/export/dump/MK.zip \
    https://download.geonames.org/export/dump/MT.zip \
    https://download.geonames.org/export/dump/NL.zip \
    https://download.geonames.org/export/dump/NO.zip \
    https://download.geonames.org/export/dump/PL.zip \
    https://download.geonames.org/export/dump/PT.zip \
    https://download.geonames.org/export/dump/RO.zip \
    https://download.geonames.org/export/dump/RS.zip \
    https://download.geonames.org/export/dump/RU.zip \
    https://download.geonames.org/export/dump/SE.zip \
    https://download.geonames.org/export/dump/SI.zip \
    https://download.geonames.org/export/dump/SL.zip \
    https://download.geonames.org/export/dump/SK.zip \
    https://download.geonames.org/export/dump/SM.zip \
    https://download.geonames.org/export/dump/TR.zip \
    https://download.geonames.org/export/dump/UA.zip \
    https://download.geonames.org/export/dump/VA.zip \
    /tmp/geonames/

ADD https://download.geonames.org/export/dump/alternatenames/DE.zip \
    https://download.geonames.org/export/dump/alternatenames/FR.zip \
    https://download.geonames.org/export/dump/alternatenames/GR.zip \
    https://download.geonames.org/export/dump/alternatenames/HU.zip \
    https://download.geonames.org/export/dump/alternatenames/BG.zip \
    https://download.geonames.org/export/dump/alternatenames/DK.zip \
    https://download.geonames.org/export/dump/alternatenames/EE.zip \
    https://download.geonames.org/export/dump/alternatenames/CH.zip \
    https://download.geonames.org/export/dump/alternatenames/CY.zip \
    https://download.geonames.org/export/dump/alternatenames/CZ.zip \
    https://download.geonames.org/export/dump/alternatenames/FI.zip \
    https://download.geonames.org/export/dump/alternatenames/ES.zip \
    https://download.geonames.org/export/dump/alternatenames/GB.zip \
    https://download.geonames.org/export/dump/alternatenames/IT.zip \
    https://download.geonames.org/export/dump/alternatenames/LI.zip \
    https://download.geonames.org/export/dump/alternatenames/LT.zip \
    https://download.geonames.org/export/dump/alternatenames/LU.zip \
    https://download.geonames.org/export/dump/alternatenames/LV.zip \
    https://download.geonames.org/export/dump/alternatenames/ME.zip \
    https://download.geonames.org/export/dump/alternatenames/MK.zip \
    https://download.geonames.org/export/dump/alternatenames/MT.zip \
    https://download.geonames.org/export/dump/alternatenames/NL.zip \
    https://download.geonames.org/export/dump/alternatenames/NO.zip \
    https://download.geonames.org/export/dump/alternatenames/PL.zip \
    https://download.geonames.org/export/dump/alternatenames/PT.zip \
    https://download.geonames.org/export/dump/alternatenames/RO.zip \
    https://download.geonames.org/export/dump/alternatenames/RS.zip \
    https://download.geonames.org/export/dump/alternatenames/RU.zip \
    https://download.geonames.org/export/dump/alternatenames/SE.zip \
    https://download.geonames.org/export/dump/alternatenames/SI.zip \
    https://download.geonames.org/export/dump/alternatenames/SL.zip \
    https://download.geonames.org/export/dump/alternatenames/SK.zip \
    https://download.geonames.org/export/dump/alternatenames/SM.zip \
    https://download.geonames.org/export/dump/alternatenames/TR.zip \
    https://download.geonames.org/export/dump/alternatenames/UA.zip \
    https://download.geonames.org/export/dump/alternatenames/VA.zip \
    /tmp/alternateNames/

RUN mkdir -p /data/geonames /data/alternateNames && \
    for COUNTRY in DE FR GR HU BG DK EE CH CY CZ FI ES GB IT LI LT LU LV ME MK MT NL NO PL PT RO RS RU SE SI SL SK SM TR UA VA; do \
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