FROM rust:latest

ARG VERSION="latest"

WORKDIR /app/src/
RUN git clone --branch "$VERSION" "https://github.com/texttechnologylab/gazetteer-rs/" /app/src/
RUN cargo build --release 
RUN mv /app/src/target/release/gazetteer /app/

WORKDIR /app/
RUN rm -rf /app/src/
RUN chmod +x /app/gazetteer 
