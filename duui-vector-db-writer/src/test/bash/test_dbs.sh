#!/usr/bin/env bash
# Startet/stoppt wegwerfbare Postgres- (mit pgvector) und Qdrant-Instanzen
# fuer DUUIVectorDbWriterTest. Nichts wird persistiert: die Container laufen
# mit --rm, Postgres-Daten liegen im tmpfs, Qdrant ohne Volume -- nach "stop"
# ist alles weg.
#
#   src/test/bash/test_dbs.sh start   # starten und warten, bis beide bereit sind
#   src/test/bash/test_dbs.sh stop    # stoppen (und damit loeschen)
#
# Zugangsdaten und Ports entsprechen den Defaults des Tests.
set -euo pipefail

PG_CONTAINER=duui-vector-db-writer-test-postgres
QDRANT_CONTAINER=duui-vector-db-writer-test-qdrant

PG_IMAGE=${PG_IMAGE:-pgvector/pgvector:pg17}
QDRANT_IMAGE=${QDRANT_IMAGE:-qdrant/qdrant:latest}

PG_PORT=${PG_PORT:-5432}
QDRANT_PORT=${QDRANT_PORT:-6333}

PG_USER=duui
PG_PASSWORD=duui
PG_DATABASE=duui_test

start() {
  docker run -d --rm --name "$PG_CONTAINER" \
    -e POSTGRES_USER="$PG_USER" \
    -e POSTGRES_PASSWORD="$PG_PASSWORD" \
    -e POSTGRES_DB="$PG_DATABASE" \
    --tmpfs /var/lib/postgresql/data \
    -p "$PG_PORT":5432 \
    "$PG_IMAGE" > /dev/null

  docker run -d --rm --name "$QDRANT_CONTAINER" \
    -p "$QDRANT_PORT":6333 \
    "$QDRANT_IMAGE" > /dev/null

  echo -n "Waiting for Postgres "
  until docker exec "$PG_CONTAINER" pg_isready -q -U "$PG_USER" -d "$PG_DATABASE"; do
    echo -n "."
    sleep 1
  done
  echo " ready"

  echo -n "Waiting for Qdrant "
  until curl -sf "http://localhost:$QDRANT_PORT/readyz" > /dev/null; do
    echo -n "."
    sleep 1
  done
  echo " ready"

  cat <<EOF

Postgres: postgresql://localhost:$PG_PORT/$PG_DATABASE (user $PG_USER, password $PG_PASSWORD)
Qdrant:   http://localhost:$QDRANT_PORT

Start the writer (it reaches the databases via host.docker.internal):

  src/test/bash/test_writer.sh start

Run the tests:

  mvn test -Dtest=DUUIVectorDbWriterTest \\
    -Dpg.connection=postgresql://host.docker.internal:$PG_PORT/$PG_DATABASE \\
    -Dqdrant.port=$QDRANT_PORT

Stop (deletes all data): $0 stop
EOF
}

stop() {
  docker stop "$PG_CONTAINER" "$QDRANT_CONTAINER" > /dev/null 2>&1 || true
  echo "Stopped and removed $PG_CONTAINER and $QDRANT_CONTAINER"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  *)
    echo "Usage: $0 start|stop" >&2
    exit 1
    ;;
esac
