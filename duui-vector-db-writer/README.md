# DUUI Vector-DB-Writer

Ein DUUI-Tool zum Schreiben von Embedding-Annotationen aus dem CAS in eine
Vektordatenbank. Liest `org.texttechnologylab.uima.type.Embedding`-Annotationen
(z.B. erzeugt von [duui-sentence-transformers](../duui-sentence-transformers)),
schreibt sie samt Modellname und Satz-Offsets in eine Zieltabelle und
protokolliert den Schreibvorgang als `DocumentModification`-Annotation zurück
in den CAS. Verändert keine fachlichen Annotationen — reines Sink-Tool.

**Status:** Beide Backends sind implementiert — `postgres` (pgvector) und
`qdrant`. Welches Backend genutzt wird, entscheidet der Parameter
`db_backend` pro Aufruf; derselbe laufende Container kann beide bedienen,
solange die jeweiligen Zugangsdaten beim Start als Env-Variablen mitgegeben
wurden.

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

## Use as Stand-Alone-Image

```sh
docker run -p 9714:9714 \
  -e DUUI_VECTOR_DB_WRITER_DB_HOST=<host> \
  -e DUUI_VECTOR_DB_WRITER_DB_PORT=5432 \
  -e DUUI_VECTOR_DB_WRITER_DB_NAME=<db> \
  -e DUUI_VECTOR_DB_WRITER_DB_USER=<user> \
  -e DUUI_VECTOR_DB_WRITER_DB_PASSWORD=<password> \
  -e DUUI_VECTOR_DB_WRITER_QDRANT_HOST=<host> \
  -e DUUI_VECTOR_DB_WRITER_QDRANT_PORT=6333 \
  docker.texttechnologylab.org/duui-vector-db-writer:latest
```

Nur eines der beiden Backends braucht wirklich erreichbare Zugangsdaten —
das jeweils andere wird erst beim tatsächlichen Aufruf mit diesem
`db_backend`-Wert kontaktiert.

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
                .withParameter("target_table_prefix", "embeddings")
);

composer.run(cas);
```

## Parameter

| Parameter | Pflicht | Beschreibung |
| --- | --- | --- |
| `db_backend` | ja | `postgres` oder `qdrant` |
| `target_table` | genau eines von beiden | Alle Modelle schreiben in dieselbe Tabelle/Collection (Embedding-Dimension muss übereinstimmen) |
| `target_table_prefix` | genau eines von beiden | Pro Modell wird eine eigene Tabelle/Collection `<prefix>_<sanitizer_modellname>` angelegt |

Bei Qdrant heißt `target_table`/`target_table_prefix` inhaltlich "Collection"
statt "Tabelle" — der Parametername ist bewusst backend-neutral gehalten,
damit ein Aufrufer nicht wissen muss, welches Backend gerade dahinter steckt.

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

Pro Dokument wird zusätzlich zu den Satz-Embeddings (`agg = NONE`) je eine
spaltenweise Mean/Min/Max-Aggregation über alle Sätze des Dokuments
geschrieben (`agg = MEAN|MIN|MAX`) — passend zum offenen Arbeitspaket
"Gesamtdurchschnitt je Experiment mit Min und Max".

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

Distanzmetrik ist `EUCLID`, nicht das sonst bei Sentence-Embeddings übliche
`COSINE` — bewusst so gewählt, damit sie zur euklidischen Distanzberechnung
passt, die der Rest der `audio-nlp-pipeline` bereits nutzt
(`SentenceDistanceAnalyzer`). Die Punkt-ID wird deterministisch aus den
fachlichen Schlüsselfeldern abgeleitet (UUID5), ein erneuter Schreibvorgang
überschreibt denselben Punkt per Upsert statt ihn zu duplizieren — das
Postgres-Gegenstück dazu ist `ON CONFLICT DO NOTHING`.

## Required UIMA input

```java
org.texttechnologylab.uima.type.Embedding
de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData   // optional, sonst doc_id="unknown_document"
```

## UIMA output

```java
org.texttechnologylab.annotation.DocumentModification   // Protokoll des Schreibvorgangs
```

## Herkunft

Tabellen-/Aggregationsschema angelehnt an eine bestehende lokale
Java-`AnalysisEngine` (`DUUIPostgresEmbeddingWriter`, siehe
`../../demos/README.md` im audio-nlp-pipeline-Projekt), hier neu als
eigenständiges Docker/REST-DUUI-Tool umgesetzt statt als lokale
UIMA-Komponente, damit es wie die übrigen Tools in diesem Repo aus jeder
DUUI-Pipeline heraus ansprechbar ist.

# Cite

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]
