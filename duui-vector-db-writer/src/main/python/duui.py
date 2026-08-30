import logging
import re
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import Any, List, Optional
import uuid

import numpy as np
import psycopg2
from cassis import load_typesystem
from fastapi import FastAPI, Response, Depends, Body, HTTPException
from fastapi.responses import PlainTextResponse
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    db_host: str
    db_port: int = 5432
    db_name: str
    db_user: str
    db_password: str

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None

    class Config:
        env_prefix = 'duui_vector_db_writer_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Vector-DB-Writer")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

# Der Writer liest Embedding-Annotationen aus dem CAS (produziert z.B. von
# duui-sentence-transformers) und schreibt sie in eine Vektordatenbank.
# Er reichert den CAS nicht mit neuen fachlichen Annotationen an, sondern nur
# mit einem Protokolleintrag (DocumentModification) -- reines Sink-Tool.
TEXTIMAGER_ANNOTATOR_INPUT_TYPES = [
    "org.texttechnologylab.uima.type.Embedding"
]

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    "org.texttechnologylab.annotation.DocumentModification"
]

SUPPORTED_BACKENDS = ["postgres", "qdrant"]

# PostgreSQL-Identifier koennen nicht parametrisiert werden (kein Platzhalter
# in DDL), deshalb nur gegen diese Whitelist geprueften Namen erlauben.
IDENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_IDENTIFIER_LENGTH = 63


class EmbeddingIn(BaseModel):
    begin: int
    end: int
    vector: List[float]
    model_name: str


class ProcessRequest(BaseModel):
    doc_id: str
    db_backend: str
    target_table: Optional[str] = None
    target_table_prefix: Optional[str] = None
    embeddings: List[EmbeddingIn]


class ProcessResponse(BaseModel):
    status: str
    written: int = 0
    table: Optional[str] = None
    comment: str
    timestamp: int


class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: Optional[str]
    meta: Optional[dict]
    supported_backends: List[str]


class TextImagerInputOutput(BaseModel):
    inputs: List[str]
    outputs: List[str]


typesystem_filename = 'src/main/resources/TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")

lua_communication_script_filename = "src/main/lua/communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")

app = FastAPI(
    title=settings.annotator_name,
    description="TTLab TextImager DUUI Vector-DB-Writer",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    return TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
        },
        supported_backends=SUPPORTED_BACKENDS,
    )


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem_xml_content,
        media_type="application/xml"
    )


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(
        inputs=TEXTIMAGER_ANNOTATOR_INPUT_TYPES,
        outputs=TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES
    )


def get_process_request(body: Any = Body(...)) -> ProcessRequest:
    try:
        if isinstance(body, (bytes, str)):
            return ProcessRequest.model_validate_json(body)
        return ProcessRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_input=False))


# Postgres-Verbindung wird beim ersten Request lazy aufgebaut und offen
# gehalten (analog zum Modell-Caching bei duui-sentence-transformers), damit
# nicht pro Satz/Dokument neu verbunden werden muss.
_pg_conn = None
_pg_lock = Lock()
_pg_known_tables = set()


def get_pg_connection():
    global _pg_conn
    with _pg_lock:
        if _pg_conn is None or _pg_conn.closed:
            _pg_conn = psycopg2.connect(
                host=settings.db_host,
                port=settings.db_port,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
            )
            _pg_conn.autocommit = False
            with _pg_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            _pg_conn.commit()
            register_vector(_pg_conn)
            logger.info("Connected to Postgres at %s:%d/%s", settings.db_host, settings.db_port, settings.db_name)
        return _pg_conn


def sanitize_model_name(model_name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
    s = re.sub(r"^_+|_+$", "", s)
    return s.lower()


def table_name_for(prefix: str, model_name: str) -> str:
    if not IDENT_PATTERN.match(prefix):
        raise ValueError(f"Invalid table prefix: {prefix}")
    sanitized = sanitize_model_name(model_name)
    if not sanitized:
        raise ValueError(f"Model name sanitizes to empty identifier: {model_name}")
    table_name = f"{prefix}_{sanitized}"
    if not IDENT_PATTERN.match(table_name):
        raise ValueError(f"Derived table name is not a valid SQL identifier: {table_name}")
    if len(table_name) > MAX_IDENTIFIER_LENGTH:
        truncated = table_name[:MAX_IDENTIFIER_LENGTH]
        logger.warning(
            "Table name for model \"%s\" is %d characters and exceeds Postgres' %d-character "
            "identifier limit; using \"%s\" (the server would truncate to exactly this anyway).",
            model_name, len(table_name), MAX_IDENTIFIER_LENGTH, truncated
        )
        return truncated
    return table_name


def ensure_table(conn, table_name: str, embedding_dim: int) -> None:
    if table_name in _pg_known_tables:
        return
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} "
            f"(id TEXT NOT NULL, model TEXT NOT NULL, begin_offset INTEGER NOT NULL, "
            f"end_offset INTEGER NOT NULL, agg TEXT NOT NULL, embedding vector({embedding_dim}) NOT NULL, "
            f"PRIMARY KEY (id, model, begin_offset, end_offset, agg))"
        )
    conn.commit()
    _pg_known_tables.add(table_name)
    logger.info("Ensured table %s (dim %d)", table_name, embedding_dim)


def resolve_target_name(request: ProcessRequest) -> str:
    """Gemeinsam fuer beide Backends: Tabellen-/Collection-Name aus
    target_table (exakt) oder target_table_prefix (+ Modellname) ableiten.
    Wirft ValueError bei ungueltiger/fehlender Angabe."""
    has_full = bool(request.target_table)
    has_prefix = bool(request.target_table_prefix)
    if has_full == has_prefix:
        raise ValueError("Exactly one of target_table or target_table_prefix must be set")

    model_name = request.embeddings[0].model_name
    if has_full:
        if not IDENT_PATTERN.match(request.target_table):
            raise ValueError(f"Invalid table name: {request.target_table}")
        return request.target_table
    return table_name_for(request.target_table_prefix, model_name)


def write_postgres(request: ProcessRequest) -> ProcessResponse:
    now = int(time())

    if not request.embeddings:
        return ProcessResponse(status="ok", written=0, comment="No embeddings in document, nothing written", timestamp=now)

    # Aktuell wird angenommen, dass alle Embeddings eines Requests vom selben
    # Modell stammen (so wie es duui-sentence-transformers heute produziert).
    model_name = request.embeddings[0].model_name
    embedding_dim = len(request.embeddings[0].vector)

    try:
        table_name = resolve_target_name(request)
    except ValueError as e:
        return ProcessResponse(status="error", comment=str(e), timestamp=now)

    conn = get_pg_connection()
    try:
        ensure_table(conn, table_name, embedding_dim)

        vectors = np.array([e.vector for e in request.embeddings], dtype=np.float32)
        rows = []
        for e in request.embeddings:
            rows.append((request.doc_id, model_name, e.begin, e.end, "NONE", np.array(e.vector, dtype=np.float32)))

        first_begin = request.embeddings[0].begin
        last_end = request.embeddings[-1].end
        rows.append((request.doc_id, model_name, first_begin, last_end, "MEAN", vectors.mean(axis=0)))
        rows.append((request.doc_id, model_name, first_begin, last_end, "MIN", vectors.min(axis=0)))
        rows.append((request.doc_id, model_name, first_begin, last_end, "MAX", vectors.max(axis=0)))

        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {table_name} (id, model, begin_offset, end_offset, agg, embedding) "
                f"VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                rows,
            )
        conn.commit()

        return ProcessResponse(
            status="ok",
            written=len(rows),
            table=table_name,
            comment=f"Wrote {len(rows)} rows ({len(request.embeddings)} sentences + mean/min/max) to {table_name}",
            timestamp=now,
        )
    except Exception as ex:
        conn.rollback()
        logger.exception(ex)
        return ProcessResponse(status="error", comment=f"Postgres write failed: {ex}", timestamp=now)


_qdrant_client = None
_qdrant_lock = Lock()
_qdrant_known_collections = set()


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    with _qdrant_lock:
        if _qdrant_client is None:
            _qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
            )
            logger.info("Connected to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port)
        return _qdrant_client


def ensure_collection(client: QdrantClient, collection_name: str, embedding_dim: int) -> None:
    if collection_name in _qdrant_known_collections:
        return
    try:
        client.get_collection(collection_name)
    except (UnexpectedResponse, ValueError):
        # Euklidische Distanz, damit sie zur restlichen Pipeline passt
        # (SentenceDistanceAnalyzer rechnet ebenfalls euklidisch).
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=embedding_dim, distance=qmodels.Distance.EUCLID),
        )
        logger.info("Created Qdrant collection %s (dim %d)", collection_name, embedding_dim)
    _qdrant_known_collections.add(collection_name)


def _point_id(doc_id: str, model_name: str, begin: int, end: int, agg: str) -> str:
    # Qdrant-Punkte brauchen eine UUID oder einen unsigned Integer als ID.
    # Deterministisch aus den fachlichen Schluesselfeldern ableiten, damit ein
    # erneuter Schreibvorgang denselben Punkt per Upsert ueberschreibt statt
    # dupliziert (Gegenstueck zu ON CONFLICT DO NOTHING bei Postgres).
    key = f"{doc_id}|{model_name}|{begin}|{end}|{agg}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def write_qdrant(request: ProcessRequest) -> ProcessResponse:
    now = int(time())

    if not request.embeddings:
        return ProcessResponse(status="ok", written=0, comment="No embeddings in document, nothing written", timestamp=now)

    model_name = request.embeddings[0].model_name
    embedding_dim = len(request.embeddings[0].vector)

    try:
        collection_name = resolve_target_name(request)
    except ValueError as e:
        return ProcessResponse(status="error", comment=str(e), timestamp=now)

    try:
        client = get_qdrant_client()
        ensure_collection(client, collection_name, embedding_dim)

        vectors = np.array([e.vector for e in request.embeddings], dtype=np.float32)
        points = []
        for e in request.embeddings:
            points.append(qmodels.PointStruct(
                id=_point_id(request.doc_id, model_name, e.begin, e.end, "NONE"),
                vector=e.vector,
                payload={"id": request.doc_id, "model": model_name, "begin_offset": e.begin, "end_offset": e.end, "agg": "NONE"},
            ))

        first_begin = request.embeddings[0].begin
        last_end = request.embeddings[-1].end
        for agg, vector in (
            ("MEAN", vectors.mean(axis=0)),
            ("MIN", vectors.min(axis=0)),
            ("MAX", vectors.max(axis=0)),
        ):
            points.append(qmodels.PointStruct(
                id=_point_id(request.doc_id, model_name, first_begin, last_end, agg),
                vector=vector.tolist(),
                payload={"id": request.doc_id, "model": model_name, "begin_offset": first_begin, "end_offset": last_end, "agg": agg},
            ))

        client.upsert(collection_name=collection_name, points=points)

        return ProcessResponse(
            status="ok",
            written=len(points),
            table=collection_name,
            comment=f"Wrote {len(points)} points ({len(request.embeddings)} sentences + mean/min/max) to {collection_name}",
            timestamp=now,
        )
    except Exception as ex:
        logger.exception(ex)
        return ProcessResponse(status="error", comment=f"Qdrant write failed: {ex}", timestamp=now)


@app.post("/v1/process")
def post_process(request: ProcessRequest = Depends(get_process_request)) -> ProcessResponse:
    if request.db_backend == "postgres":
        response = write_postgres(request)
    elif request.db_backend == "qdrant":
        response = write_qdrant(request)
    else:
        response = ProcessResponse(
            status="error",
            comment=f"Unknown db_backend \"{request.db_backend}\", expected one of {SUPPORTED_BACKENDS}",
            timestamp=int(time()),
        )

    logger.info(response.comment)
    return response
