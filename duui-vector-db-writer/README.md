# DUUI Vector-DB-Writer

Ein DUUI-Tool zum Schreiben von Embedding-Annotationen aus dem CAS in eine
Vektordatenbank. Liest `org.texttechnologylab.uima.type.Embedding`-Annotationen
(z.B. erzeugt von [duui-sentence-transformers](../duui-sentence-transformers)),
schreibt sie samt Modellname und Satz-Offsets in eine Zieltabelle und
protokolliert den Schreibvorgang als `DocumentModification`-Annotation zurück
in den CAS. Verändert keine fachlichen Annotationen — reines Sink-Tool.

**Status:** Beide Backends sind implementiert — `postgres` (pgvector) und
`qdrant`. Welches Backend genutzt wird, entscheidet der Parameter
`db_backend` pro Aufruf; derselbe laufende Container kann beide bedienen.
Wie bei den übrigen DUUI-Tools wird alles, inkl. Zugangsdaten, per
`withParameter` übergeben — der Container selbst braucht keine
Konfiguration.

## Voraussetzungen

Für das Postgres-Backend: eine erreichbare PostgreSQL-Instanz mit
installierter [pgvector](https://github.com/pgvector/pgvector)-Extension
(die Extension wird beim ersten Connect automatisch angelegt, falls die
DB-Rolle das darf):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Für das Qdrant-Backend: eine erreichbare [Qdrant](https://qdrant.tech/)-Instanz
(z.B. `docker run -p 6333:6333 qdrant/qdrant`) — Collections werden beim
ersten Schreibvorgang automatisch angelegt.

**Wichtig:** Die Datenbank wird aus dem Container heraus angesprochen. Host
und Port müssen also von dort erreichbar sein — `localhost` meint im
Container den Container selbst, nicht den Docker-Host. Läuft die Datenbank
auf dem Docker-Host, ist sie unter `host.docker.internal` erreichbar; unter
Linux ist dieser Name nicht automatisch definiert und muss beim Start mit
`--add-host=host.docker.internal:host-gateway` angelegt werden (Docker
Desktop unter macOS/Windows kennt ihn von selbst).

## Use as Stand-Alone-Image

```sh
docker run -p 9714:9714 docker.texttechnologylab.org/duui-vector-db-writer:latest

# Linux, Datenbank auf dem Docker-Host:
docker run -p 9714:9714 --add-host=host.docker.internal:host-gateway \
  docker.texttechnologylab.org/duui-vector-db-writer:latest
```

## Run within DUUI using previously started docker container

```java
DUUIComposer composer = new DUUIComposer()
        .withLuaContext(
                new DUUILuaContext()
                        .withJsonLibrary()
        ).withSkipVerification(true);
DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
composer.addDriver(remote_driver);
composer.add(
        new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
                .withParameter("db_backend", "postgres")
                .withParameter("pg_connection_string", "postgresql://<host>:5432/<db>")
                .withParameter("pg_user", "<user>")
                .withParameter("pg_password", "<password>")
                .withParameter("target_table_prefix", "embeddings")
);

composer.run(cas);
```

## Run within DUUI using the Docker driver

```java
DUUIDockerDriver docker_driver = new DUUIDockerDriver();
composer.addDriver(docker_driver);
composer.add(
        new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-vector-db-writer:0.0.2")
                .withParameter("db_backend", "postgres")
                .withParameter("pg_connection_string", "postgresql://<host>:<port>/<db>")
                .withParameter("pg_user", "<user>")
                .withParameter("pg_password", "<password>")
                .withParameter("target_table", "<table>")
);
```

Qdrant:

```java
composer.add(
        new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
                .withParameter("db_backend", "qdrant")
                .withParameter("target_table_prefix", "embeddings")
                .withParameter("qdrant_url", "https://xyz.cloud.qdrant.io:6333")
                .withParameter("qdrant_api_key", "<api-key>")
);
```

## Parameter

| Parameter | Pflicht | Beschreibung |
| --- | --- | --- |
| `db_backend` | ja | `postgres` oder `qdrant` |
| `target_table` | genau eines von beiden | Alle Modelle schreiben in dieselbe Tabelle/Collection (Embedding-Dimension muss übereinstimmen) |
| `target_table_prefix` | genau eines von beiden | Pro Modell wird eine eigene Tabelle/Collection `<prefix>_<modellname>` angelegt (siehe [Namen](#tabellen--und-collection-namen)). Leerer Wert (`""`): ohne Präfix, nur `<modellname>` — z.B. für eine DB, die ausschließlich Embeddings enthält |
| `fail_on_error` | nein | `true` (Default): ein fehlgeschlagener Schreibvorgang liefert HTTP 400/500, DUUI bricht das Dokument ab. `false`: Fehler wird nur als `DocumentModification` (Kommentar `Error: ...`) im CAS protokolliert, die Pipeline läuft weiter |

Postgres (`db_backend=postgres`):

| Parameter | Pflicht | Beschreibung |
| --- | --- | --- |
| `pg_connection_string` | entweder dies … | libpq-Connection-String, z.B. `postgresql://host:port/database`. Hat Vorrang vor `pg_host`/`pg_port`/`pg_database` |
| `pg_host`, `pg_database` | … oder diese | Host und Datenbankname |
| `pg_port` | nein | Default `5432` |
| `pg_user` | ja (ohne Connection-String) | DB-User |
| `pg_password` | nein | DB-Passwort |

Qdrant (`db_backend=qdrant`):

| Parameter | Pflicht | Beschreibung |
| --- | --- | --- |
| `qdrant_url` | entweder dies … | z.B. `https://xyz.cloud.qdrant.io:6333`; hat Vorrang vor `qdrant_host`/`qdrant_port`. Ohne Port in der URL gilt der Default des Schemas (443/80) |
| `qdrant_host` | … oder dies | Host |
| `qdrant_port` | nein | Default `6333` |
| `qdrant_api_key` | nein | API-Key |
| `qdrant_distance` | nein | `cosine` (Default), `euclid`, `dot` oder `manhattan` — siehe [Collection-Schema (Qdrant)](#collection-schema-qdrant) für die Erklärung, wann welches Maß sinnvoll ist |

**Hinweis:** Als Parameter übergebene Passwörter/API-Keys stehen im Klartext
in der Composer-Konfiguration und können je nach Setup von DUUI mitgespeichert
werden. Der Service selbst loggt sie nicht.

**Fehlerverhalten:** Fehlt im CAS die `DocumentMetaData` (bzw. deren
`documentId`) oder hat ein Embedding keine `modelReference` mit `source`,
schlägt das Dokument immer fehl (unabhängig von `fail_on_error`) — die
Dokument-ID und der Modellname sind Teil des Schlüssels, ein Platzhalter würde
Dokumente gegenseitig überschreiben. Soll die Pipeline bei fehlgeschlagenen
Dokumenten weiterlaufen, `DUUIComposer.withIgnoreErrors(true)` setzen.

`DUUIComposer.run(JCas)` legt für einen CAS ganz ohne `DocumentMetaData`
selbst eine an, mit `documentId = "UIMA-Document"`. Diese ID hätten alle
solchen Dokumente gemeinsam, sie würden sich gegenseitig überschreiben — der
Writer lehnt sie deshalb ebenfalls ab. Die Dokument-ID muss also immer vom
Reader bzw. Aufrufer gesetzt werden.

Enthält ein CAS Embeddings mehrerer Modelle (z.B. zwei
duui-sentence-transformers-Läufe), wird pro Modell gruppiert: mit
`target_table_prefix` landet jedes Modell in seiner eigenen
Tabelle/Collection, mit `target_table` alle in derselben (dann müssen die
Dimensionen übereinstimmen, sonst Fehler). Mean/Min/Max werden je Modell
berechnet. Bei Postgres werden alle Modelle eines Dokuments in einer
Transaktion geschrieben.

Bei Qdrant heißt `target_table`/`target_table_prefix` inhaltlich "Collection"
statt "Tabelle" — der Parametername ist bewusst backend-neutral gehalten,
damit ein Aufrufer nicht wissen muss, welches Backend gerade dahinter steckt.

## Tabellen- und Collection-Namen

Mit `target_table_prefix` wird der Name aus Präfix und Modellname gebildet,
je nach Backend unterschiedlich — der Modellname selbst steht in beiden Fällen
unverändert in der Spalte bzw. im Payload-Feld `model`:

| Backend | Regel | Beispiel (`embeddings` + `nomic-ai/nomic-embed-text-v2-moe`) |
| --- | --- | --- |
| `postgres` | Nicht-alphanumerische Zeichen werden zu `_`, alles kleingeschrieben (Postgres faltet unquotierte Namen ohnehin auf Kleinschreibung, so bleibt die Tabelle auch unquotiert ansprechbar). Länger als 63 Zeichen wird abgeschnitten | `embeddings_nomic_ai_nomic_embed_text_v2_moe` |
| `qdrant` | Modellname bleibt erhalten, nur die von Qdrant verbotenen Zeichen `< > : " / \ \| ? *` werden zu `__`. Maximal 255 Zeichen | `embeddings_nomic-ai__nomic-embed-text-v2-moe` |

Mit leerem `target_table_prefix` entfällt `embeddings_` im Beispiel, der Name
ist dann nur `nomic_ai_nomic_embed_text_v2_moe` bzw.
`nomic-ai__nomic-embed-text-v2-moe`. Postgres-Tabellennamen müssen mit einem
Buchstaben oder `_` beginnen; beginnt der Modellname mit einer Ziffer, schlägt
der Schreibvorgang in dem Fall mit einem Hinweis fehl, ein Präfix zu setzen.

Ein explizites `target_table` wird bei Postgres als SQL-Identifier
(`[A-Za-z_][A-Za-z0-9_]*`, max. 63 Zeichen) geprüft und kleingeschrieben, bei
Qdrant nur gegen die verbotenen Zeichen geprüft.

## Tabellenschema (Postgres)

```sql
CREATE TABLE <table> (
    id           TEXT NOT NULL,       -- Dokument-ID (DocumentMetaData)
    model        TEXT NOT NULL,
    begin_offset INTEGER NOT NULL,
    end_offset   INTEGER NOT NULL,
    agg          TEXT NOT NULL,       -- NONE (ein Satz) | MEAN | MIN | MAX (Dokumentaggregat)
    embedding    vector(<dim>) NOT NULL,
    PRIMARY KEY (id, model, begin_offset, end_offset, agg)
);
```

Pro Dokument und Modell wird zusätzlich zu den Satz-Embeddings (`agg = NONE`)
je eine spaltenweise Mean/Min/Max-Aggregation über alle Sätze des Dokuments
geschrieben (`agg = MEAN|MIN|MAX`). Diese Zeilen decken das ganze Dokument ab:
`begin_offset` ist der Begin des ersten, `end_offset` das End des letzten
Embeddings.

## Collection-Schema (Qdrant)

Gleiches Datenmodell wie bei Postgres, nur als Payload statt als Spalten:

```json
{
  "id": "point-uuid (deterministisch aus id+model+begin+end+agg)",
  "vector": [...],
  "payload": {
    "id": "<Dokument-ID>",
    "model": "<Modellname>",
    "begin_offset": 0,
    "end_offset": 42,
    "agg": "NONE"
  }
}
```

Distanzmetrik ist per `qdrant_distance` wählbar:

| Wert | Bedeutung | Wann benutzen |
| --- | --- | --- |
| `cosine` (Default) | Winkel zwischen zwei Vektoren, Länge des Vektors spielt keine Rolle | Bei den meisten Sentence-Embedding-Modellen trägt die Richtung des Vektors die Bedeutung, nicht die Länge — deshalb der gebräuchlichste Standard für "welche Sätze sind sich inhaltlich am ähnlichsten", z.B. für semantische Suche |
| `euclid` | Geometrischer Abstand zwischen zwei Punkten (Pythagoras) | Wenn Ergebnisse mit einer anderen, euklidisch rechnenden Auswertung vergleichbar sein sollen |
| `dot` | Skalarprodukt (Kombination aus Winkel und Länge) | Wenn die Vektorlänge selbst Information trägt (z.B. bei manchen unnormalisierten Embeddings) oder für Performance, wenn Vektoren schon normalisiert sind (dann ist `dot` rechnerisch gleichwertig zu `cosine`, aber schneller) |
| `manhattan` | Abstand entlang der Achsen statt diagonal (Summe der Betragsdifferenzen) | Seltener bei Embeddings; eher relevant, wenn einzelne Dimensionen unabhängig voneinander interpretiert werden sollen |

**Wichtig:** Qdrant legt das Distanzmaß beim Anlegen der Collection fest und
erlaubt danach **keine Änderung mehr**. Existiert die Ziel-Collection schon
mit einem anderen Maß als angefragt, antwortet der Writer mit einem Fehler
statt das Maß stillschweigend zu ignorieren — in dem Fall entweder eine neue
Collection (anderer `target_table`/`target_table_prefix`) verwenden oder die
bestehende vorher löschen.

Die Punkt-ID wird deterministisch aus den fachlichen Schlüsselfeldern
abgeleitet (UUID5), ein erneuter Schreibvorgang überschreibt denselben Punkt
per Upsert statt ihn zu duplizieren. Postgres verhält sich gleich
(`ON CONFLICT ... DO UPDATE SET embedding = EXCLUDED.embedding`): in beiden
Backends gewinnt der zuletzt geschriebene Vektor.

## Testen

```sh
src/main/bash/docker_build.sh            # Image bauen
src/test/bash/test_dbs.sh start          # wegwerfbare Postgres- (pgvector) und Qdrant-Container
src/test/bash/test_writer.sh start       # Writer-Container (mit --add-host fuer host.docker.internal)
mvn test -Dtest=DUUIVectorDbWriterTest
src/test/bash/test_writer.sh stop
src/test/bash/test_dbs.sh stop           # entfernt die DBs samt aller Daten
```

Die DB-Container nutzen die Defaults von `DUUIVectorDbWriterTest`, Ports und
Images sind per `PG_PORT`/`QDRANT_PORT`/`PG_IMAGE`/`QDRANT_IMAGE` bzw.
`WRITER_PORT`/`WRITER_IMAGE` änderbar.

## Required UIMA input

```java
org.texttechnologylab.uima.type.Embedding
de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData   // Pflicht (documentId), sonst schlägt das Dokument fehl
```

## UIMA output

```java
org.texttechnologylab.annotation.DocumentModification   // Protokoll des Schreibvorgangs (Tool-Name, Version, Ergebnis)
```

# Cite

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]
