#!/usr/bin/env bash
# Startet/stoppt den Vector-DB-Writer-Container fuer DUUIVectorDbWriterTest.
# Der Container erreicht die Datenbanken auf dem Docker-Host (z.B. aus
# test_dbs.sh) ueber host.docker.internal -- unter Linux muss der Name per
# --add-host angelegt werden, sonst schlaegt jeder Schreibvorgang mit
# "could not translate host name" fehl.
#
#   src/test/bash/test_writer.sh start   # starten und warten, bis der Service antwortet
#   src/test/bash/test_writer.sh stop    # stoppen (und damit loeschen)
#
# Das Image vorher mit src/main/bash/docker_build.sh bauen.
set -euo pipefail

WRITER_CONTAINER=duui-vector-db-writer-test
WRITER_IMAGE=${WRITER_IMAGE:-docker.texttechnologylab.org/duui-vector-db-writer:latest}
WRITER_PORT=${WRITER_PORT:-9714}

start() {
  if ! docker image inspect "$WRITER_IMAGE" > /dev/null 2>&1; then
    echo "Image $WRITER_IMAGE not found, build it first: src/main/bash/docker_build.sh" >&2
    exit 1
  fi

  docker run -d --rm --name "$WRITER_CONTAINER" \
    --add-host=host.docker.internal:host-gateway \
    -p "$WRITER_PORT":9714 \
    "$WRITER_IMAGE" > /dev/null

  echo -n "Waiting for writer "
  until curl -sf "http://localhost:$WRITER_PORT/v1/documentation" > /dev/null; do
    if ! docker container inspect "$WRITER_CONTAINER" > /dev/null 2>&1; then
      echo " container exited, check the image (docker run $WRITER_IMAGE)" >&2
      exit 1
    fi
    echo -n "."
    sleep 1
  done
  echo " ready"

  cat <<EOF

Writer: http://localhost:$WRITER_PORT
Logs:   docker logs -f $WRITER_CONTAINER
Stop:   $0 stop
EOF
}

stop() {
  docker stop "$WRITER_CONTAINER" > /dev/null 2>&1 || true
  echo "Stopped and removed $WRITER_CONTAINER"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  *)
    echo "Usage: $0 start|stop" >&2
    exit 1
    ;;
esac
