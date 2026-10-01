import logging
import re
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from platform import python_version
from sys import version as sys_version
from threading import BoundedSemaphore, Lock
from time import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import uuid

import numpy as np
import psycopg2
from cassis import load_typesystem
from fastapi import FastAPI, Response, Depends, Body, HTTPException
from fastapi.responses import PlainTextResponse
from pgvector.psycopg2 import register_vector
from psycopg2 import sql
from psycopg2.extensions import parse_dsn
from psycopg2.pool import ThreadedConnectionPool
from pydantic import BaseModel, SecretStr, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

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
PG_IDENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PG_MAX_IDENTIFIER_LENGTH = 63

# Qdrant legt jede Collection als Verzeichnis an und verbietet deshalb diese
# Zeichen (qdrant: lib/common/common/src/validation.rs, INVALID_NAME_CHARS),
# ausserdem "." / ".." und mehr als 255 Zeichen. Alles andere bleibt so, wie
# es im Modellnamen steht.
QDRANT_INVALID_NAME_CHARS = set('<>:"/\\|?*\0\x1f')
QDRANT_INVALID_NAME_REPLACEMENT = "__"
QDRANT_MAX_NAME_LENGTH = 255

AGGREGATIONS = ("MEAN", "MIN", "MAX")


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
    qdrant_distance: Optional[str] = None
    embeddings: List[EmbeddingIn]

    # Verbindungsdaten kommen wie bei allen DUUI-Tools ausschliesslich als
    # Parameter (withParameter), nicht ueber Env-Variablen des Containers.
    # Backend-spezifische Parameter sind mit pg_ bzw. qdrant_ praefixiert.
    pg_connection_string: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: int = 5432
    pg_database: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[SecretStr] = None

    qdrant_url: Optional[str] = None
    qdrant_host: Optional[str] = None
    qdrant_port: int = 6333
    qdrant_api_key: Optional[SecretStr] = None

    # true: Schreibfehler fuehren zu HTTP 4xx/5xx, DUUI bricht das Dokument ab.
    # false: Fehler wird nur als DocumentModification im CAS protokolliert.
    fail_on_error: bool = True

    @field_validator("target_table_prefix", mode="before")
    @classmethod
    def empty_prefix_means_no_prefix(cls, value: Any) -> Any:
        # Anders als bei den uebrigen Parametern ist "" hier ein gueltiger
        # Wert: eine Tabelle/Collection pro Modell, nur nach dem Modell
        # benannt. Nicht gesetzt (None) heisst: Parameter nicht angegeben.
        if isinstance(value, str) and value.strip() == "":
            return ""
        return value

    @field_validator(
        "target_table", "qdrant_distance",
        "pg_connection_string", "pg_host", "pg_port", "pg_database", "pg_user", "pg_password",
        "qdrant_url", "qdrant_host", "qdrant_port", "qdrant_api_key",
        mode="before",
    )
    @classmethod
    def empty_string_is_unset(cls, value: Any, info: ValidationInfo) -> Any:
        # DUUI-Parameter sind immer Strings; ein leerer Wert bedeutet "nicht
        # gesetzt" und faellt auf den Default des Feldes zurueck.
        if isinstance(value, str) and value.strip() == "":
            return cls.model_fields[info.field_name].default
        return value


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class ProcessResponse(BaseModel):
    status: str
    written: int = 0
    tables: List[str] = []
    modification_meta: DocumentModification


class TextImagerCapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: Optional[str]
    meta: Optional[dict]
    docker_container_id: Optional[str]
    parameters: Optional[dict]
    capability: TextImagerCapability
    implementation_specific: Optional[str]


PARAMETER_DOCUMENTATION = {
    "db_backend": "Required: one of " + ", ".join(SUPPORTED_BACKENDS),
    "target_table": "Exact table/collection name for all models (exactly one of target_table/target_table_prefix)",
    "target_table_prefix": "One table/collection per model: <prefix>_<model_name>; empty: only <model_name>",
    "fail_on_error": "true (default): failed writes return HTTP 4xx/5xx; false: only log the error in the CAS",
    "pg_connection_string": "postgresql://host:port/database; overrides pg_host/pg_port/pg_database",
    "pg_host": "Postgres host (if no pg_connection_string)",
    "pg_port": "Postgres port, default 5432",
    "pg_database": "Postgres database (if no pg_connection_string)",
    "pg_user": "Postgres user",
    "pg_password": "Postgres password",
    "qdrant_url": "Qdrant URL, e.g. https://xyz.cloud.qdrant.io:6333; overrides qdrant_host/qdrant_port",
    "qdrant_host": "Qdrant host (if no qdrant_url)",
    "qdrant_port": "Qdrant port, default 6333",
    "qdrant_api_key": "Qdrant API key",
    "qdrant_distance": "cosine (default), euclid, dot or manhattan",
}


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


def _library_version(*distributions: str) -> Optional[str]:
    # Mehrere Distributionsnamen, da z.B. psycopg2 auch als "psycopg2"
    # statt "psycopg2-binary" installiert sein kann.
    for distribution in distributions:
        try:
            return package_version(distribution)
        except PackageNotFoundError:
            continue
    return None


# Versionen der DB-Client-Bibliotheken, damit nachvollziehbar ist, womit
# geschrieben wurde. Einmal beim Start bestimmt, aendert sich zur Laufzeit nicht.
LIBRARY_VERSIONS = {
    "psycopg2": _library_version("psycopg2-binary", "psycopg2"),
    "pgvector": _library_version("pgvector"),
    "qdrant_client": _library_version("qdrant-client"),
    "numpy": _library_version("numpy"),
}


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    return TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "library_versions": LIBRARY_VERSIONS,
        },
        docker_container_id=None,
        parameters=PARAMETER_DOCUMENTATION,
        capability=TextImagerCapability(
            supported_languages=[],
            reproducible=True,
        ),
        implementation_specific="Supported backends: " + ", ".join(SUPPORTED_BACKENDS),
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


def _secret(value: Optional[SecretStr]) -> Optional[str]:
    return value.get_secret_value() if value is not None else None


def _missing_message(backend: str, missing: List[str]) -> str:
    return f"Missing {backend} connection parameter(s): {', '.join(missing)}"


# ---------------------------------------------------------------------------
# Gemeinsame Aufbereitung fuer beide Backends
# ---------------------------------------------------------------------------

class ModelBatch(NamedTuple):
    model_name: str
    target: str
    embeddings: List[EmbeddingIn]
    vectors: np.ndarray

    def entries(self) -> List[Tuple[int, int, str, np.ndarray]]:
        """Satz-Embeddings (agg=NONE) plus spaltenweise Mean/Min/Max ueber
        alle Saetze dieses Modells im Dokument. Die Aggregate decken das ganze
        Dokument ab: Begin des ersten bis End des letzten Embeddings (in
        CAS-Reihenfolge)."""
        rows = [(e.begin, e.end, "NONE", self.vectors[i]) for i, e in enumerate(self.embeddings)]
        doc_begin = self.embeddings[0].begin
        doc_end = self.embeddings[-1].end
        for agg, vector in zip(AGGREGATIONS, (
            self.vectors.mean(axis=0),
            self.vectors.min(axis=0),
            self.vectors.max(axis=0),
        )):
            rows.append((doc_begin, doc_end, agg, vector))
        return rows


# --- Postgres-Tabellennamen -------------------------------------------------
# Namen werden kleingeschrieben: Aufrufer (z.B. ein Reader, der bereits
# geschriebene Dokumente ueberspringt) sprechen die Tabelle typischerweise
# unquotiert an, und unquotierte Identifier faltet Postgres auf Kleinschreibung.
# Ein gequoteter Name mit Grossbuchstaben waere fuer sie nicht auffindbar.

def pg_sanitize_model_name(model_name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
    s = re.sub(r"^_+|_+$", "", s)
    return s.lower()


def pg_table_name(request: ProcessRequest, model_name: str) -> str:
    if request.target_table:
        if not PG_IDENT_PATTERN.match(request.target_table):
            raise ValueError(f"Invalid Postgres table name: {request.target_table}")
        if len(request.target_table) > PG_MAX_IDENTIFIER_LENGTH:
            raise ValueError(
                f"Table name \"{request.target_table}\" exceeds Postgres' "
                f"{PG_MAX_IDENTIFIER_LENGTH}-character identifier limit"
            )
        return request.target_table.lower()

    prefix = request.target_table_prefix
    if prefix and not PG_IDENT_PATTERN.match(prefix):
        raise ValueError(f"Invalid Postgres table prefix: {prefix}")
    sanitized = pg_sanitize_model_name(model_name)
    if not sanitized:
        raise ValueError(f"Model name sanitizes to empty identifier: {model_name}")
    table_name = f"{prefix}_{sanitized}".lower() if prefix else sanitized
    if not PG_IDENT_PATTERN.match(table_name):
        # Nur ohne Praefix moeglich: der Modellname beginnt mit einer Ziffer.
        raise ValueError(
            f"Table name \"{table_name}\" for model \"{model_name}\" is not a valid Postgres identifier "
            f"(must start with a letter or _); set a non-empty target_table_prefix"
        )
    if len(table_name) > PG_MAX_IDENTIFIER_LENGTH:
        truncated = table_name[:PG_MAX_IDENTIFIER_LENGTH]
        logger.warning(
            "Table name for model \"%s\" is %d characters and exceeds Postgres' %d-character "
            "identifier limit; using \"%s\". Check that no other model truncates to the same name.",
            model_name, len(table_name), PG_MAX_IDENTIFIER_LENGTH, truncated
        )
        return truncated
    return table_name


# --- Qdrant-Collectionnamen -------------------------------------------------
# So nah wie moeglich am Original: nur die von Qdrant verbotenen Zeichen
# werden ersetzt, z.B. "nomic-ai/nomic-embed-text-v2-moe" ->
# "<prefix>_nomic-ai__nomic-embed-text-v2-moe".

def qdrant_validate_name(name: str, what: str) -> str:
    if not name or name in (".", ".."):
        raise ValueError(f"Invalid Qdrant {what}: \"{name}\"")
    invalid = sorted(set(name) & QDRANT_INVALID_NAME_CHARS)
    if invalid:
        raise ValueError(f"Qdrant {what} \"{name}\" contains forbidden characters: {invalid}")
    if len(name) > QDRANT_MAX_NAME_LENGTH:
        raise ValueError(f"Qdrant {what} \"{name}\" exceeds {QDRANT_MAX_NAME_LENGTH} characters")
    return name


def qdrant_collection_name(request: ProcessRequest, model_name: str) -> str:
    if request.target_table:
        return qdrant_validate_name(request.target_table, "collection name")

    model_part = "".join(
        QDRANT_INVALID_NAME_REPLACEMENT if c in QDRANT_INVALID_NAME_CHARS else c for c in model_name
    )
    prefix = request.target_table_prefix
    if not prefix:
        return qdrant_validate_name(model_part, "collection name")
    qdrant_validate_name(prefix, "collection prefix")
    return qdrant_validate_name(f"{prefix}_{model_part}", "collection name")


TARGET_NAMERS = {
    "postgres": pg_table_name,
    "qdrant": qdrant_collection_name,
}


def resolve_target_name(request: ProcessRequest, model_name: str) -> str:
    """Tabellen-/Collection-Name aus target_table (exakt) oder
    target_table_prefix (+ Modellname) nach den Regeln des Backends ableiten.
    Wirft ValueError bei ungueltiger/fehlender Angabe."""
    # target_table_prefix="" ist gesetzt (kein Praefix), nur None ist "fehlt".
    if bool(request.target_table) == (request.target_table_prefix is not None):
        raise ValueError("Exactly one of target_table or target_table_prefix must be set")
    return TARGET_NAMERS[request.db_backend](request, model_name)


def prepare_batches(request: ProcessRequest) -> List[ModelBatch]:
    """Embeddings nach Modell gruppieren (ein CAS kann Embeddings mehrerer
    sentence-transformers-Laeufe enthalten) und je Modell Ziel und Vektoren
    bestimmen. Mehrfach vorhandene Spannen desselben Modells (z.B. Modell
    zweimal gelaufen) werden auf das zuletzt gelesene Embedding reduziert."""
    groups: Dict[str, Dict[Tuple[int, int], EmbeddingIn]] = {}
    for e in request.embeddings:
        groups.setdefault(e.model_name, {})[(e.begin, e.end)] = e

    batches = []
    for model_name, by_span in groups.items():
        embeddings = list(by_span.values())
        dims = {len(e.vector) for e in embeddings}
        if len(dims) != 1:
            raise ValueError(f"Embeddings of model \"{model_name}\" have inconsistent dimensions: {sorted(dims)}")
        if 0 in dims:
            raise ValueError(f"Embeddings of model \"{model_name}\" are empty")
        vectors = np.array([e.vector for e in embeddings], dtype=np.float32)
        batches.append(ModelBatch(model_name, resolve_target_name(request, model_name), embeddings, vectors))

    dims_by_target: Dict[str, Dict[str, int]] = {}
    for batch in batches:
        dims_by_target.setdefault(batch.target, {})[batch.model_name] = batch.vectors.shape[1]
    for target, dims in dims_by_target.items():
        if len(set(dims.values())) > 1:
            raise ValueError(
                f"Models with different embedding dimensions would be written to the same target \"{target}\" "
                f"({dims}); use target_table_prefix to get one table/collection per model"
            )
    return batches


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

class PgConfig(NamedTuple):
    dsn: Optional[str]
    host: Optional[str]
    port: int
    dbname: Optional[str]
    user: Optional[str]
    password: Optional[str]

    def connect_kwargs(self) -> Dict[str, Any]:
        if self.dsn:
            kwargs: Dict[str, Any] = {"dsn": self.dsn}
        else:
            kwargs = {"host": self.host, "port": self.port, "dbname": self.dbname}
        # user/password ergaenzen bzw. ueberschreiben die Werte im DSN.
        if self.user is not None:
            kwargs["user"] = self.user
        if self.password is not None:
            kwargs["password"] = self.password
        return kwargs

    def describe(self) -> str:
        # Bewusst ohne User/Passwort, damit es gefahrlos geloggt werden kann.
        return f"{self.host}:{self.port}/{self.dbname}"


def resolve_pg_config(request: ProcessRequest) -> PgConfig:
    user = request.pg_user
    password = _secret(request.pg_password)

    if request.pg_connection_string:
        # libpq-Connection-String, z.B. postgresql://host:port/database
        dsn = request.pg_connection_string
        if request.pg_host or request.pg_database:
            logger.warning("pg_connection_string is set, ignoring pg_host/pg_port/pg_database")
        try:
            parsed = parse_dsn(dsn)
        except psycopg2.ProgrammingError:
            # Fehlermeldung von libpq nicht weitergeben, sie kann Teile des
            # Strings (inkl. Passwort) enthalten.
            raise ValueError("Invalid pg_connection_string, expected postgresql://host:port/database") from None
        return PgConfig(
            dsn=dsn,
            host=parsed.get("host"),
            port=int(parsed.get("port", 5432)),
            dbname=parsed.get("dbname"),
            user=user,
            password=password,
        )

    missing = [name for name, value in (
        ("pg_host", request.pg_host), ("pg_database", request.pg_database), ("pg_user", user)
    ) if value is None]
    if missing:
        raise ValueError(_missing_message("Postgres", missing + ["(or pg_connection_string)"]))
    return PgConfig(
        dsn=None,
        host=request.pg_host,
        port=request.pg_port,
        dbname=request.pg_database,
        user=user,
        password=password,
    )


class _VectorConnection(psycopg2.extensions.connection):
    # Merkt sich, ob der pgvector-Typ fuer diese Verbindung schon registriert ist.
    vector_registered = False


# Maximal so viele gleichzeitige Verbindungen pro Ziel-DB; weitere Requests
# warten, statt dass der Pool mit PoolError abbricht.
PG_POOL_MAX_CONNECTIONS = 8


class PgPool:
    """Verbindungspool pro Verbindungskonfiguration. Jeder Request bekommt
    eine eigene Verbindung und damit eine eigene Transaktion -- eine einzige
    geteilte Verbindung wuerde bei parallelen DUUI-Workern Commits/Rollbacks
    verschiedener Dokumente vermischen."""

    def __init__(self, config: PgConfig):
        self._pool = ThreadedConnectionPool(
            1, PG_POOL_MAX_CONNECTIONS, connection_factory=_VectorConnection, **config.connect_kwargs()
        )
        self._slots = BoundedSemaphore(PG_POOL_MAX_CONNECTIONS)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        except Exception:
            self._pool.closeall()
            raise
        self._pool.putconn(conn)

    @contextmanager
    def connection(self):
        with self._slots:
            conn = self._pool.getconn()
            broken = False
            try:
                if not conn.vector_registered:
                    register_vector(conn)
                    conn.vector_registered = True
                yield conn
                conn.commit()
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                # Verbindung ist tot (z.B. DB-Neustart): verwerfen statt
                # zurueck in den Pool, der naechste Request baut neu auf.
                broken = True
                raise
            except Exception:
                conn.rollback()
                raise
            finally:
                self._pool.putconn(conn, close=broken or bool(conn.closed))


_pg_pools: Dict[PgConfig, PgPool] = {}
_pg_pools_lock = Lock()
_pg_known_tables = set()
# CREATE TABLE IF NOT EXISTS ist bei parallelen Aufrufen nicht race-frei
# (pg_type unique violation). Der Lock serialisiert innerhalb des Prozesses,
# der Advisory-Lock in ensure_table zusaetzlich ueber Prozesse/Container
# hinweg (uvicorn --workers > 1, DUUI withScale > 1).
_pg_ddl_lock = Lock()


def get_pg_pool(config: PgConfig) -> PgPool:
    with _pg_pools_lock:
        pool = _pg_pools.get(config)
        if pool is None:
            pool = PgPool(config)
            _pg_pools[config] = pool
            logger.info("Connected to Postgres at %s", config.describe())
        return pool


def ensure_table(conn, config: PgConfig, table_name: str, embedding_dim: int) -> None:
    cache_key = (config, table_name)
    with _pg_ddl_lock:
        if cache_key in _pg_known_tables:
            return
        with conn.cursor() as cur:
            # Wird mit dem commit unten automatisch wieder freigegeben.
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (table_name,))
            cur.execute(sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} "
                "(id TEXT NOT NULL, model TEXT NOT NULL, begin_offset INTEGER NOT NULL, "
                "end_offset INTEGER NOT NULL, agg TEXT NOT NULL, embedding vector({}) NOT NULL, "
                "PRIMARY KEY (id, model, begin_offset, end_offset, agg))"
            ).format(sql.Identifier(table_name), sql.Literal(embedding_dim)))
        conn.commit()
        _pg_known_tables.add(cache_key)
    logger.info("Ensured table %s (dim %d)", table_name, embedding_dim)


def write_postgres(request: ProcessRequest, batches: List[ModelBatch]) -> Tuple[int, List[str]]:
    config = resolve_pg_config(request)
    pool = get_pg_pool(config)

    tables = [batch.target for batch in batches]

    written = 0
    with pool.connection() as conn:
        for batch, table_name in zip(batches, tables):
            ensure_table(conn, config, table_name, batch.vectors.shape[1])

        # Alle Modelle eines Dokuments in einer Transaktion: ganz oder gar nicht.
        with conn.cursor() as cur:
            for batch, table_name in zip(batches, tables):
                rows = [
                    (request.doc_id, batch.model_name, begin, end, agg, vector)
                    for begin, end, agg, vector in batch.entries()
                ]
                cur.executemany(
                    sql.SQL(
                        "INSERT INTO {} (id, model, begin_offset, end_offset, agg, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (id, model, begin_offset, end_offset, agg) "
                        "DO UPDATE SET embedding = EXCLUDED.embedding"
                    ).format(sql.Identifier(table_name)),
                    rows,
                )
                written += len(rows)
    return written, tables


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

class QdrantConfig(NamedTuple):
    url: Optional[str]
    host: Optional[str]
    port: Optional[int]
    api_key: Optional[str]

    def describe(self) -> str:
        return self.url if self.url else f"{self.host}:{self.port}"


def resolve_qdrant_config(request: ProcessRequest) -> QdrantConfig:
    api_key = _secret(request.qdrant_api_key)
    if request.qdrant_url:
        return QdrantConfig(url=request.qdrant_url, host=None, port=None, api_key=api_key)
    if not request.qdrant_host:
        raise ValueError(_missing_message("Qdrant", ["qdrant_url or qdrant_host"]))
    return QdrantConfig(url=None, host=request.qdrant_host, port=request.qdrant_port, api_key=api_key)


_qdrant_clients: Dict[QdrantConfig, QdrantClient] = {}
_qdrant_clients_lock = Lock()
# (Client-Konfiguration, Collection) -> (Dimension, Distanzmass) der
# tatsaechlich existierenden Collection, damit ein wiederholter Aufruf mit
# abweichendem qdrant_distance auch dann noch erkannt wird, wenn die
# Collection in diesem Prozess schon einmal bestaetigt wurde.
_qdrant_known_collections: Dict[Tuple[QdrantConfig, str], Tuple[int, qmodels.Distance]] = {}
_qdrant_ddl_lock = Lock()

# Qdrant legt das Distanzmass beim Anlegen der Collection fest und erlaubt
# danach keine Aenderung mehr (dafuer muesste die Collection neu angelegt
# werden) -- deshalb pro Aufruf waehlbar. Default cosine, das uebliche Mass
# fuer Sentence-Embeddings.
QDRANT_DISTANCE_MAP = {
    "euclid": qmodels.Distance.EUCLID,
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "manhattan": qmodels.Distance.MANHATTAN,
}
DEFAULT_QDRANT_DISTANCE = "cosine"


def resolve_qdrant_distance(name: Optional[str]) -> qmodels.Distance:
    key = (name or DEFAULT_QDRANT_DISTANCE).lower()
    if key not in QDRANT_DISTANCE_MAP:
        raise ValueError(
            f"Unknown qdrant_distance \"{name}\", expected one of {sorted(QDRANT_DISTANCE_MAP)}"
        )
    return QDRANT_DISTANCE_MAP[key]


def get_qdrant_client(config: QdrantConfig) -> QdrantClient:
    with _qdrant_clients_lock:
        client = _qdrant_clients.get(config)
        if client is None:
            if config.url:
                # port=None: ohne Port in der URL gilt der Default des Schemas
                # (443/80) statt des qdrant-client-Defaults 6333.
                client = QdrantClient(url=config.url, port=None, api_key=config.api_key)
            else:
                client = QdrantClient(host=config.host, port=config.port, api_key=config.api_key)
            _qdrant_clients[config] = client
            logger.info("Connected to Qdrant at %s", config.describe())
        return client


def _collection_params(client: QdrantClient, collection_name: str) -> Tuple[int, qmodels.Distance]:
    vectors = client.get_collection(collection_name).config.params.vectors
    if not isinstance(vectors, qmodels.VectorParams):
        raise ValueError(
            f"Collection \"{collection_name}\" uses named vectors, which this writer does not support"
        )
    return vectors.size, vectors.distance


def ensure_collection(client: QdrantClient, config: QdrantConfig, collection_name: str,
                      embedding_dim: int, distance: qmodels.Distance) -> None:
    cache_key = (config, collection_name)
    with _qdrant_ddl_lock:
        known = _qdrant_known_collections.get(cache_key)
        if known is None:
            if client.collection_exists(collection_name):
                known = _collection_params(client, collection_name)
            else:
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=qmodels.VectorParams(size=embedding_dim, distance=distance),
                    )
                    logger.info("Created Qdrant collection %s (dim %d, distance %s)",
                                collection_name, embedding_dim, distance.value)
                    known = (embedding_dim, distance)
                except UnexpectedResponse as ex:
                    # 409: ein anderer Prozess/Container hat die Collection
                    # zwischen collection_exists und create_collection angelegt.
                    if ex.status_code != 409:
                        raise
                    known = _collection_params(client, collection_name)
            _qdrant_known_collections[cache_key] = known

    existing_dim, existing_distance = known
    if existing_distance != distance:
        raise ValueError(
            f"Collection \"{collection_name}\" already exists with distance "
            f"\"{existing_distance.value}\", requested \"{distance.value}\" cannot be "
            f"applied retroactively -- use a different target_table/target_table_prefix "
            f"or delete the existing collection first"
        )
    if existing_dim != embedding_dim:
        raise ValueError(
            f"Collection \"{collection_name}\" already exists with dimension {existing_dim}, "
            f"embeddings have dimension {embedding_dim}"
        )


def _point_id(doc_id: str, model_name: str, begin: int, end: int, agg: str) -> str:
    # Qdrant-Punkte brauchen eine UUID oder einen unsigned Integer als ID.
    # Deterministisch aus den fachlichen Schluesselfeldern ableiten, damit ein
    # erneuter Schreibvorgang denselben Punkt per Upsert ueberschreibt statt
    # dupliziert (gleiches Verhalten wie ON CONFLICT DO UPDATE bei Postgres).
    key = f"{doc_id}|{model_name}|{begin}|{end}|{agg}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def write_qdrant(request: ProcessRequest, batches: List[ModelBatch]) -> Tuple[int, List[str]]:
    distance = resolve_qdrant_distance(request.qdrant_distance)
    config = resolve_qdrant_config(request)
    client = get_qdrant_client(config)

    written = 0
    for batch in batches:
        ensure_collection(client, config, batch.target, batch.vectors.shape[1], distance)
        points = [
            qmodels.PointStruct(
                id=_point_id(request.doc_id, batch.model_name, begin, end, agg),
                vector=vector.tolist(),
                payload={"id": request.doc_id, "model": batch.model_name,
                         "begin_offset": begin, "end_offset": end, "agg": agg},
            )
            for begin, end, agg, vector in batch.entries()
        ]
        client.upsert(collection_name=batch.target, points=points)
        written += len(points)
    return written, [batch.target for batch in batches]


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

WRITERS = {
    "postgres": write_postgres,
    "qdrant": write_qdrant,
}


def build_response(status: str, message: str, timestamp: int, written: int = 0,
                   tables: Optional[List[str]] = None) -> ProcessResponse:
    # Wie bei duui-sentence-transformers: Name und Version des Tools stehen im
    # DocumentModification, damit im CAS nachvollziehbar ist, was geschrieben hat.
    return ProcessResponse(
        status=status,
        written=written,
        tables=tables or [],
        modification_meta=DocumentModification(
            user=settings.annotator_name,
            timestamp=timestamp,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), {message}",
        ),
    )


@app.post("/v1/process")
def post_process(request: ProcessRequest = Depends(get_process_request)) -> ProcessResponse:
    now = int(time())
    backend = request.db_backend

    try:
        writer = WRITERS.get(backend)
        if writer is None:
            raise ValueError(f"Unknown db_backend \"{backend}\", expected one of {SUPPORTED_BACKENDS}")

        if not request.embeddings:
            response = build_response("ok", "No embeddings in document, nothing written", now)
        else:
            batches = prepare_batches(request)
            written, tables = writer(request, batches)
            sentences = sum(len(batch.embeddings) for batch in batches)
            response = build_response(
                "ok",
                f"Wrote {written} entries ({sentences} sentences + mean/min/max for "
                f"{len(batches)} model(s)) to {backend}: {', '.join(tables)}",
                now,
                written=written,
                tables=tables,
            )
        logger.info(response.modification_meta.comment)
        return response

    except ValueError as ex:
        status_code = 400
        message = str(ex)
    except Exception as ex:
        logger.exception(ex)
        status_code = 500
        message = f"{backend} write failed: {ex}"

    logger.error(message)
    if request.fail_on_error:
        raise HTTPException(status_code=status_code, detail=message)
    return build_response("error", f"Error: {message}", now)
