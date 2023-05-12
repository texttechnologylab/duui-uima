ARG VERSION="latest"

FROM docker.texttechnologylab.org/gazetteer-rs/base:${VERSION}
WORKDIR /app

COPY config.toml /app/
COPY communication_layer.lua /app/
COPY resources/ /app/resources/

EXPOSE 9714

ENTRYPOINT ["/app/gazetteer", "--config", "config.toml", "--address", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1", "--limit", "536870912"]
